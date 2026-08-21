from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import d4_capture_support as support  # noqa: E402
import d4_linux  # noqa: E402


def _topology() -> str:
    rows = ["# CPU,Core,Socket,Online"]
    rows.extend(f"{cpu},{cpu},0,Y" for cpu in range(8))
    rows.extend(f"{cpu + 8},{cpu},0,Y" for cpu in range(8))
    return "\n".join(rows)


def _limits(*, cpu_quota: int = 800_000, cpuset: str = "0-15") -> dict[str, str]:
    return {
        "/sys/fs/cgroup/cpu.max": f"{cpu_quota} 100000",
        "/sys/fs/cgroup/cpuset.cpus.effective": cpuset,
        "/sys/fs/cgroup/memory.max": str(16 * 1024**3),
    }


def _unavailable_power_policy() -> dict[str, dict[str, str]]:
    return {
        "paths": {},
        "unavailable": {
            group: "unavailable in test sysfs"
            for group in support.LOCAL_LINUX_POWER_POLICY_SUFFIXES
        },
    }


def _probe(thread_count: int, *, cpu: int = 3, ratio: float = 1.0) -> str:
    wall_seconds = 1.01
    return json.dumps(
        {
            "native_thread_count": thread_count,
            "wall_seconds": wall_seconds,
            "process_cpu_seconds": wall_seconds * ratio,
            "process_cpu_wall_ratio": ratio,
            "worker_reports": [
                {
                    "worker_index": index,
                    "native_thread_id": 1000 + index,
                    "affinity": [cpu],
                    "observed_cpus": [cpu],
                    "iterations": 100,
                }
                for index in range(thread_count)
            ],
        }
    )


class D4LocalLinuxTests(unittest.TestCase):
    def test_baseline_requires_native_linux_x86(self) -> None:
        d4_linux.assert_linux_baseline(
            system="Linux", machine="x86_64", cpu_description="Intel Core Ultra"
        )
        with self.assertRaisesRegex(support.D4CaptureError, "requires Linux"):
            d4_linux.assert_linux_baseline(
                system="Darwin", machine="x86_64", cpu_description="Intel"
            )
        with self.assertRaisesRegex(support.D4CaptureError, "native x86_64"):
            d4_linux.assert_linux_baseline(system="Linux", machine="aarch64", cpu_description="ARM")
        with self.assertRaisesRegex(support.D4CaptureError, "emulation marker"):
            d4_linux.assert_linux_baseline(
                system="Linux", machine="x86_64", cpu_description="QEMU Virtual CPU"
            )

    def test_attestation_binds_complete_cgroup_and_physical_masks(self) -> None:
        visible = list(range(16))
        masks = support.physical_core_masks(_topology(), allowed_cpus=visible)
        attestation = d4_linux.linux_resource_attestation(
            cgroup_limits=_limits(),
            lscpu_parse=_topology(),
            visible_affinity=visible,
            masks=masks,
            memory_total_bytes=32 * 1024**3,
            power_policy=_unavailable_power_policy(),
            dedicated_capture=True,
            stable=True,
        )
        self.assertEqual(attestation["mode"], "cgroup_v2_cpuset")
        self.assertEqual(attestation["allocated_physical_cores"], 8)
        self.assertEqual(attestation["allocated_memory_bytes"], 16 * 1024**3)
        self.assertEqual(attestation["physical_core_masks"]["t8"], list(range(8)))
        self.assertFalse(attestation["exclusive_physical_cores"])
        self.assertTrue(attestation["dedicated_capture"])

    def test_attestation_rejects_cpuset_quota_and_operator_control_drift(self) -> None:
        visible = list(range(16))
        masks = support.physical_core_masks(_topology(), allowed_cpus=visible)
        common = {
            "lscpu_parse": _topology(),
            "visible_affinity": visible,
            "masks": masks,
            "memory_total_bytes": 32 * 1024**3,
            "power_policy": _unavailable_power_policy(),
            "dedicated_capture": True,
            "stable": True,
        }
        with self.assertRaisesRegex(support.D4CaptureError, "quota"):
            d4_linux.linux_resource_attestation(cgroup_limits=_limits(cpu_quota=700_000), **common)
        with self.assertRaisesRegex(support.D4CaptureError, "cpuset"):
            d4_linux.linux_resource_attestation(cgroup_limits=_limits(cpuset="0-14"), **common)
        with self.assertRaisesRegex(support.D4CaptureError, "dedicated-capture and stable"):
            d4_linux.linux_resource_attestation(
                cgroup_limits=_limits(), **{**common, "stable": False}
            )

    def test_power_policy_snapshot_records_values_or_explicit_unavailability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            governor = root / "cpu0/cpufreq/scaling_governor"
            governor.parent.mkdir(parents=True)
            governor.write_text("performance\n", encoding="utf-8")
            snapshot = d4_linux._power_policy_snapshot([0], root=root)
        self.assertEqual(snapshot["paths"][str(governor)], "performance")
        self.assertNotIn("scaling_governor", snapshot["unavailable"])
        self.assertEqual(
            set(snapshot["unavailable"]),
            set(support.LOCAL_LINUX_POWER_POLICY_SUFFIXES) - {"scaling_governor"},
        )

    def test_affinity_preflight_runs_frozen_native_thread_matrix_under_t1(self) -> None:
        with mock.patch.object(
            d4_linux,
            "_capture",
            side_effect=[_probe(count) for count in support.LOCAL_LINUX_AFFINITY_PROBE_THREADS],
        ) as capture:
            pre = d4_linux._affinity_enforcement_phase([3], environment={})
        self.assertEqual(
            [probe["native_thread_count"] for probe in pre["probes"]],
            list(support.LOCAL_LINUX_AFFINITY_PROBE_THREADS),
        )
        for call in capture.call_args_list:
            command = call.args[0]
            self.assertEqual(command[:3], ["taskset", "--cpu-list", "3"])
        controls = d4_linux._affinity_enforcement(expected_affinity=[3], pre=pre, post=pre)
        validated = support.validate_local_linux_affinity_enforcement(
            controls, expected_affinity=[3]
        )
        self.assertEqual(validated, controls)

    def test_affinity_preflight_rejects_cpu_wall_oversubscription(self) -> None:
        with (
            mock.patch.object(d4_linux, "_capture", return_value=_probe(1, ratio=1.3)),
            self.assertRaisesRegex(support.D4CaptureError, "frozen controls"),
        ):
            d4_linux._affinity_enforcement_phase([3], environment={})

    def test_build_plan_separates_venvs_and_keeps_native_override_supplemental(self) -> None:
        masks = {budget: tuple(range(budget)) for budget in support.THREAD_BUDGETS}
        plan = d4_linux.build_plan(Path("/capture"), affinity_masks=masks)
        shipping = plan["builds"]["shipping"]
        native = plan["builds"]["native"]
        self.assertNotEqual(shipping["venv"], native["venv"])
        self.assertNotIn("target-cpu=native", " ".join(shipping["build"]))
        self.assertIn("RUSTFLAGS=-C target-cpu=native", native["build"])
        self.assertEqual(plan["provider"], "local_linux")
        self.assertEqual(plan["affinity_masks"]["t8"], list(range(8)))

    def test_compact_build_plan_uses_shipping_t1_t8_only(self) -> None:
        masks = {budget: tuple(range(budget)) for budget in (1, 8)}
        plan = d4_linux.compact_build_plan(Path("/capture"), affinity_masks=masks)
        self.assertEqual(list(plan["builds"]), ["shipping"])
        self.assertEqual(plan["thread_budgets"], [1, 8])
        self.assertEqual(plan["subprocess_count"], 84)
        self.assertEqual(plan["affinity_masks"], {"t1": [0], "t8": list(range(8))})

    def test_failed_workspace_is_retained_and_successful_workspace_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            failure = Path(temporary) / "failed"
            failure.mkdir()
            with (
                mock.patch.object(d4_linux.tempfile, "mkdtemp", return_value=str(failure)),
                self.assertRaisesRegex(RuntimeError, "boom"),
                d4_linux._capture_workspace(),
            ):
                raise RuntimeError("boom")
            self.assertTrue(failure.is_dir())

            success = Path(temporary) / "success"
            success.mkdir()
            with (
                mock.patch.object(d4_linux.tempfile, "mkdtemp", return_value=str(success)),
                d4_linux._capture_workspace(),
            ):
                (success / "evidence.txt").write_text("ok", encoding="utf-8")
            self.assertFalse(success.exists())


if __name__ == "__main__":
    unittest.main()
