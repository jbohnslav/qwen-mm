from __future__ import annotations

import sys
import tempfile
import unittest
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
    return "# CPU,Core,Socket,Online\n" + "\n".join(f"{cpu},{cpu},0,Y" for cpu in range(16))


class _Stream:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def read(self) -> str:
        return self.value


class _Process:
    def __init__(self, exit_code: int) -> None:
        self.exit_code = exit_code
        self.stdout = _Stream("stdout")
        self.stderr = _Stream("stderr")

    def wait(self) -> int:
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
    def test_plan_fixes_request_limits_runtime_lifecycle_and_pricing(self) -> None:
        plan = modal_d4._sandbox_plan(source={}, assets={}, source_revision="a" * 40)
        self.assertEqual(plan["runner"], modal_d4.SANDBOX_RUNTIME)
        self.assertEqual(plan["resources"]["cpu_request_and_hard_limit"], [16.0, 16.0])
        self.assertEqual(plan["resources"]["memory_request_and_hard_limit_mib"], [32768, 32768])
        self.assertTrue(plan["resources"]["vm_runtime"])
        self.assertTrue(plan["resources"]["nonpreemptible"])
        self.assertEqual(plan["pricing_snapshot"]["nonpreemptible_multiplier"], 1.0)
        self.assertAlmostEqual(
            plan["pricing_snapshot"]["requested_resource_usd_per_hour"], 3.039, places=3
        )

    def test_vm_attestation_requires_exact_kvm_cpuset_and_smt_free_topology(self) -> None:
        masks = {budget: tuple(range(budget)) for budget in modal_d4.THREAD_BUDGETS}
        result = modal_d4._vm_resource_attestation(
            cgroup_limits={"/sys/fs/cgroup/cpuset.cpus.effective": "0-15"},
            virtualization="kvm",
            visible_affinity=list(range(16)),
            masks=masks,
            topology=_topology(),
            control=_control(),
            observed_image_id="im-pinned",
        )
        self.assertEqual(result["mode"], modal_d4.SANDBOX_RUNTIME)
        self.assertEqual(result["resolved_modal_image_id"], "im-pinned")
        without_runtime_env = modal_d4._vm_resource_attestation(
            cgroup_limits={"/sys/fs/cgroup/cpuset.cpus.effective": "0-15"},
            virtualization="kvm",
            visible_affinity=list(range(16)),
            masks=masks,
            topology=_topology(),
            control=_control(),
            observed_image_id=None,
        )
        self.assertEqual(without_runtime_env["resolved_modal_image_id"], "im-pinned")
        with self.assertRaisesRegex(modal_d4.D4CaptureError, "runtime image"):
            modal_d4._vm_resource_attestation(
                cgroup_limits={"/sys/fs/cgroup/cpuset.cpus.effective": "0-15"},
                virtualization="kvm",
                visible_affinity=list(range(16)),
                masks=masks,
                topology=_topology(),
                control=_control(),
                observed_image_id="im-different",
            )

        cases = (
            ({"/sys/fs/cgroup/cpuset.cpus.effective": "0-14"}, "kvm", list(range(16))),
            ({"/sys/fs/cgroup/cpuset.cpus.effective": "0-15"}, "none", list(range(16))),
            ({"/sys/fs/cgroup/cpuset.cpus.effective": "0-15"}, "kvm", list(range(15))),
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
            f"{cpu},{cpu % 8},0,Y" for cpu in range(16)
        )
        smt_masks = modal_d4.physical_core_masks(smt_topology, allowed_cpus=list(range(16)))
        with self.assertRaisesRegex(modal_d4.D4CaptureError, "SMT-free"):
            modal_d4._vm_resource_attestation(
                cgroup_limits={"/sys/fs/cgroup/cpuset.cpus.effective": "0-15"},
                virtualization="kvm",
                visible_affinity=list(range(16)),
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
            payload, lifecycle = modal_d4._execute_in_vm_sandbox(
                source={}, assets={}, revision="a" * 40, probe_only=True
            )
        self.assertEqual(payload, b"probe")
        self.assertTrue(sandbox.terminated)
        self.assertTrue(sandbox.detached)
        self.assertTrue(lifecycle["terminated"])
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["cpu"], (16.0, 16.0))
        self.assertEqual(kwargs["memory"], (32768, 32768))
        self.assertEqual(kwargs["experimental_options"], {"vm_runtime": True})

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


if __name__ == "__main__":
    unittest.main()
