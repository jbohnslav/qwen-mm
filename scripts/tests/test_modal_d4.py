from __future__ import annotations

import io
import json
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import modal_d4  # noqa: E402


def _control() -> dict[str, object]:
    return {
        "runner": modal_d4.SANDBOX_RUNTIME,
        "resources": {
            "cpu_request_and_hard_limit": list(modal_d4.SANDBOX_CPU),
            "memory_request_and_hard_limit_mib": list(modal_d4.SANDBOX_MEMORY_MIB),
            "vm_runtime": True,
            "nonpreemptible": True,
            "single_use": True,
        },
        "resolved_modal_image_id": "im-pinned",
        "sandbox_id": "sb-single-use",
    }


def _topology() -> str:
    return "# CPU,Core,Socket,Online\n" + "\n".join(f"{cpu},{cpu},0,Y" for cpu in range(8))


class _Stream:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def __iter__(self):
        yield self.value

    def read(self) -> str:
        raise AssertionError("worker streams must be consumed live, before final wait collection")


class _Process:
    def __init__(self, exit_code: int) -> None:
        self.exit_code = exit_code
        self.stdout = _Stream("stdout")
        self.stderr = _Stream("stderr")

    def wait(self) -> int:
        return self.exit_code


class _DelayedProcess(_Process):
    def wait(self) -> int:
        time.sleep(0.04)
        return self.exit_code


class _Filesystem:
    def __init__(self) -> None:
        self.control = ""

    def write_text(self, value: str, path: str) -> None:
        self.control = value

    def read_bytes(self, path: str) -> bytes:
        return b"probe"


class _Sandbox:
    object_id = "sb-single-use"

    def __init__(self, exit_code: int = 0) -> None:
        self.filesystem = _Filesystem()
        self.exit_code = exit_code
        self.terminated = False
        self.detached = False

    def exec(self, *args: str, **kwargs: object) -> _Process:
        return _Process(self.exit_code)

    def terminate(self, *, wait: bool) -> int:
        if not wait:
            raise AssertionError("termination must wait")
        self.terminated = True
        return 137

    def poll(self) -> int | None:
        return 137 if self.terminated else None

    def detach(self) -> None:
        self.detached = True


class _ImageBuilder:
    object_id = "im-pinned"

    def hydrate(self) -> _ImageBuilder:
        return self


class _Image:
    @staticmethod
    def from_id(image_id: str) -> object:
        if image_id != "im-pinned":
            raise AssertionError("image identity changed")
        return object()


class ModalD4VmSandboxTests(unittest.TestCase):
    def test_image_installs_vm_virtualization_detector(self) -> None:
        self.assertIn("systemd", modal_d4.D4_APT_PACKAGES)

    def test_plan_fixes_request_limits_runtime_lifecycle_and_pricing(self) -> None:
        plan = modal_d4._sandbox_plan(source={}, assets={}, source_revision="a" * 40)
        self.assertEqual(plan["runner"], modal_d4.SANDBOX_RUNTIME)
        self.assertEqual(plan["resources"]["cpu_request_and_hard_limit"], [8.0, 8.0])
        self.assertEqual(plan["resources"]["memory_request_and_hard_limit_mib"], [16384, 16384])
        self.assertTrue(plan["resources"]["vm_runtime"])
        self.assertTrue(plan["resources"]["nonpreemptible"])
        self.assertEqual(plan["resources"]["timeout_seconds"], 14_400)
        self.assertEqual(plan["build_labels"], ["shipping"])
        self.assertEqual(plan["thread_budgets"], [1, 8])
        self.assertEqual(plan["subprocess_count"], 84)
        self.assertEqual(plan["pricing_snapshot"]["nonpreemptible_multiplier"], 1.0)
        self.assertAlmostEqual(
            plan["pricing_snapshot"]["requested_resource_usd_per_hour"], 1.52, places=2
        )
        self.assertLessEqual(
            plan["pricing_snapshot"]["hard_cost_ceiling_usd"],
            6.08,
        )

    def test_dry_run_prints_exact_plan_without_worker_creation_or_approval(self) -> None:
        def git(*arguments: str) -> str:
            if arguments == ("rev-parse", "HEAD"):
                return "a" * 40
            if arguments == ("status", "--short", "--untracked-files=all"):
                return "dirty-is-safe-for-dry-run"
            raise AssertionError(arguments)

        with (
            mock.patch.object(modal_d4, "source_payload_identity", return_value={}),
            mock.patch.object(modal_d4, "assets_identity", return_value={}),
            mock.patch.object(modal_d4, "_git", side_effect=git),
            mock.patch.object(modal_d4, "_execute_in_vm_sandbox") as execute,
            mock.patch("builtins.print") as output,
        ):
            modal_d4.main(dry_run=True)

        execute.assert_not_called()
        rendered = output.call_args.args[0]
        self.assertIn('"subprocess_count": 84', rendered)
        self.assertIn('"hard_cost_ceiling_usd": 6.077952', rendered)

    def test_vm_attestation_requires_exact_kvm_cpuset_and_smt_free_topology(self) -> None:
        masks = {budget: tuple(range(budget)) for budget in modal_d4.THREAD_BUDGETS}
        result = modal_d4._vm_resource_attestation(
            cgroup_limits={"/sys/fs/cgroup/cpuset.cpus.effective": "0-7"},
            virtualization="kvm",
            visible_affinity=list(range(8)),
            masks=masks,
            topology=_topology(),
            control=_control(),
            observed_image_id="im-pinned",
        )
        self.assertEqual(result["mode"], modal_d4.SANDBOX_RUNTIME)
        self.assertEqual(result["resolved_modal_image_id"], "im-pinned")
        without_runtime_env = modal_d4._vm_resource_attestation(
            cgroup_limits={"/sys/fs/cgroup/cpuset.cpus.effective": "0-7"},
            virtualization="kvm",
            visible_affinity=list(range(8)),
            masks=masks,
            topology=_topology(),
            control=_control(),
            observed_image_id=None,
        )
        self.assertEqual(without_runtime_env["resolved_modal_image_id"], "im-pinned")
        with self.assertRaisesRegex(modal_d4.D4CaptureError, "runtime image"):
            modal_d4._vm_resource_attestation(
                cgroup_limits={"/sys/fs/cgroup/cpuset.cpus.effective": "0-7"},
                virtualization="kvm",
                visible_affinity=list(range(8)),
                masks=masks,
                topology=_topology(),
                control=_control(),
                observed_image_id="im-different",
            )

        cases = (
            ({"/sys/fs/cgroup/cpuset.cpus.effective": "0-6"}, "kvm", list(range(8))),
            ({"/sys/fs/cgroup/cpuset.cpus.effective": "0-7"}, "none", list(range(8))),
            ({"/sys/fs/cgroup/cpuset.cpus.effective": "0-7"}, "kvm", list(range(7))),
        )
        for cgroups, virtualization, affinity in cases:
            with self.subTest(cgroups=cgroups, virtualization=virtualization, affinity=affinity):
                with self.assertRaises(modal_d4.D4CaptureError):
                    modal_d4._vm_resource_attestation(
                        cgroup_limits=cgroups,
                        virtualization=virtualization,
                        visible_affinity=affinity,
                        masks=masks,
                        topology=_topology(),
                        control=_control(),
                        observed_image_id="im-pinned",
                    )
        smt_topology = "# CPU,Core,Socket,Online\n" + "\n".join(
            f"{cpu},{cpu % 4},0,Y" for cpu in range(8)
        )
        smt_masks = modal_d4.physical_core_masks(
            smt_topology, allowed_cpus=list(range(8)), budgets=(1, 4)
        )
        with self.assertRaisesRegex(modal_d4.D4CaptureError, "SMT-free"):
            modal_d4._vm_resource_attestation(
                cgroup_limits={"/sys/fs/cgroup/cpuset.cpus.effective": "0-7"},
                virtualization="kvm",
                visible_affinity=list(range(8)),
                masks=smt_masks,
                topology=smt_topology,
                control=_control(),
                observed_image_id="im-pinned",
            )

    def test_sandbox_uses_hard_limits_and_always_terminates(self) -> None:
        sandbox = _Sandbox()
        create = mock.Mock(return_value=sandbox)
        fake_modal = mock.Mock()
        fake_modal.Image = _Image
        fake_modal.Sandbox.create = create
        with (
            mock.patch.object(modal_d4, "modal", fake_modal),
            mock.patch.object(modal_d4, "app", object()),
            mock.patch.object(modal_d4, "d4_image", _ImageBuilder()),
            mock.patch.object(modal_d4.importlib.metadata, "version", return_value="1.4.0"),
        ):
            with tempfile.TemporaryDirectory() as temporary:
                worker_log = Path(temporary) / "worker.jsonl"
                payload, lifecycle = modal_d4._execute_in_vm_sandbox(
                    source={},
                    assets={},
                    revision="a" * 40,
                    probe_only=True,
                    worker_log_path=worker_log,
                )
                log_records = [
                    json.loads(line) for line in worker_log.read_text(encoding="utf-8").splitlines()
                ]
        self.assertEqual(payload, b"probe")
        self.assertTrue(sandbox.terminated)
        self.assertTrue(sandbox.detached)
        self.assertTrue(lifecycle["terminated"])
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["cpu"], (8.0, 8.0))
        self.assertEqual(kwargs["memory"], (16384, 16384))
        self.assertEqual(kwargs["experimental_options"], {"vm_runtime": True})
        self.assertTrue(
            {("worker_output", "stdout"), ("worker_output", "stderr")}
            <= {(record.get("event"), record.get("stream")) for record in log_records}
        )

    def test_worker_emits_flushed_structured_start_and_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control_path = root / "control.json"
            output_path = root / "output.bin"
            control_path.write_text(
                json.dumps(
                    {
                        "runner": modal_d4.SANDBOX_RUNTIME,
                        "expected_source": {},
                        "expected_assets": {},
                        "source_revision": "a" * 40,
                        "modal_client_version": "1.4.0",
                        "resolved_modal_image_id": "im-pinned",
                        "sandbox_id": "sb-single-use",
                        "resources": _control()["resources"],
                    }
                ),
                encoding="utf-8",
            )
            rendered = io.StringIO()
            with (
                mock.patch.object(modal_d4, "_run_d4_capture", return_value=b"capture"),
                redirect_stdout(rendered),
            ):
                modal_d4._worker_main(
                    control_path=control_path,
                    output_path=output_path,
                    probe_only=False,
                )

        events = [json.loads(line) for line in rendered.getvalue().splitlines()]
        self.assertEqual(
            [event["stage"] for event in events], ["worker_started", "worker_completed"]
        )
        self.assertEqual(events[-1]["output_bytes"], 7)

    def test_controller_persists_heartbeats_with_the_last_worker_stage(self) -> None:
        process = _DelayedProcess(exit_code=0)
        process.stdout = _Stream('{"stage":"timed_budget_started"}\n')
        process.stderr = _Stream("")
        with tempfile.TemporaryDirectory() as temporary:
            worker_log = Path(temporary) / "worker.jsonl"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                exit_code, _stdout, _stderr = modal_d4._stream_worker_process(
                    process,
                    worker_log_path=worker_log,
                    heartbeat_seconds=0.01,
                )
            records = [
                json.loads(line) for line in worker_log.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(exit_code, 0)
        heartbeats = [record for record in records if record["event"] == "worker_heartbeat"]
        self.assertTrue(heartbeats)
        self.assertTrue(
            all(record["last_stage"] == "timed_budget_started" for record in heartbeats)
        )

    def test_make_defers_validation_to_the_locked_project_environment(self) -> None:
        makefile = (SCRIPTS.parent / "Makefile").read_text(encoding="utf-8")
        recipe = makefile.split("d4-modal-capture:\n", 1)[1].split("\nd4-linux-capture:", 1)[0]
        self.assertIn("modal run scripts/modal_d4.py", recipe)
        self.assertIn("--defer-validation", recipe)
        self.assertIn("$(UV_CMD) run --locked --no-sync --package qwen-mm-reference", recipe)
        self.assertIn("--finalize --output", recipe)

    def test_main_refuses_worker_creation_without_paid_compute_approval(self) -> None:
        with (
            mock.patch.object(modal_d4, "source_payload_identity", return_value={}),
            mock.patch.object(modal_d4, "assets_identity", return_value={}),
            mock.patch.object(modal_d4, "_git", return_value="a" * 40),
            mock.patch.object(modal_d4, "_execute_in_vm_sandbox") as execute,
            self.assertRaisesRegex(modal_d4.D4CaptureError, "approve-paid-compute"),
        ):
            modal_d4.main()
        execute.assert_not_called()

    def test_worker_failure_still_terminates_and_detaches(self) -> None:
        sandbox = _Sandbox(exit_code=12)
        fake_modal = mock.Mock()
        fake_modal.Image = _Image
        fake_modal.Sandbox.create.return_value = sandbox
        with tempfile.TemporaryDirectory() as temporary:
            lifecycle_path = Path(temporary) / "lifecycle.json"
            with (
                mock.patch.object(modal_d4, "modal", fake_modal),
                mock.patch.object(modal_d4, "app", object()),
                mock.patch.object(modal_d4, "d4_image", _ImageBuilder()),
                mock.patch.object(modal_d4.importlib.metadata, "version", return_value="1.4.0"),
                self.assertRaisesRegex(modal_d4.D4CaptureError, "worker failed"),
            ):
                modal_d4._execute_in_vm_sandbox(
                    source={},
                    assets={},
                    revision="a" * 40,
                    probe_only=True,
                    lifecycle_path=lifecycle_path,
                )
            self.assertIn('"terminated": true', lifecycle_path.read_text(encoding="utf-8"))
        self.assertTrue(sandbox.terminated)
        self.assertTrue(sandbox.detached)

    def test_main_retains_retrieved_payload_before_local_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "x86.zip"

            def git(*arguments: str) -> str:
                if arguments == ("rev-parse", "HEAD"):
                    return "a" * 40
                if arguments == ("status", "--short", "--untracked-files=all"):
                    return ""
                raise AssertionError(arguments)

            with (
                mock.patch.object(modal_d4, "source_payload_identity", return_value={}),
                mock.patch.object(modal_d4, "assets_identity", return_value={}),
                mock.patch.object(modal_d4, "assert_source_payload_matches_revision"),
                mock.patch.object(modal_d4, "_git", side_effect=git),
                mock.patch.object(modal_d4, "_sandbox_plan", return_value={}),
                mock.patch.object(
                    modal_d4,
                    "_execute_in_vm_sandbox",
                    return_value=(b"raw-zip", {"terminated": True}),
                ),
                mock.patch.object(
                    modal_d4,
                    "write_capture_archive",
                    side_effect=modal_d4.D4CaptureError("invalid archive"),
                ),
                self.assertRaisesRegex(modal_d4.D4CaptureError, "invalid archive"),
            ):
                modal_d4.main(output=str(destination), approve_paid_compute=True)

            diagnostic = destination.with_name("x86.unvalidated.zip")
            self.assertEqual(diagnostic.read_bytes(), b"raw-zip")

    def test_main_removes_unvalidated_payload_after_successful_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "x86.zip"

            def git(*arguments: str) -> str:
                if arguments == ("rev-parse", "HEAD"):
                    return "a" * 40
                if arguments == ("status", "--short", "--untracked-files=all"):
                    return ""
                raise AssertionError(arguments)

            with (
                mock.patch.object(modal_d4, "source_payload_identity", return_value={}),
                mock.patch.object(modal_d4, "assets_identity", return_value={}),
                mock.patch.object(modal_d4, "assert_source_payload_matches_revision"),
                mock.patch.object(modal_d4, "_git", side_effect=git),
                mock.patch.object(modal_d4, "_sandbox_plan", return_value={}),
                mock.patch.object(
                    modal_d4,
                    "_execute_in_vm_sandbox",
                    return_value=(b"raw-zip", {"terminated": True}),
                ),
                mock.patch.object(modal_d4, "write_capture_archive") as validate,
            ):
                modal_d4.main(output=str(destination), approve_paid_compute=True)

            validate.assert_called_once()
            self.assertFalse(destination.with_name("x86.unvalidated.zip").exists())


if __name__ == "__main__":
    unittest.main()
