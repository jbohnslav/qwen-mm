"""Summarize raw ABBA serving measurements without imposing a passing threshold."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def summarize(root):
    report = json.loads((root / "results.json").read_text())
    assert not report["failures"], report["failures"]
    assert [run["mode"] for run in report["runs"]] == ["stock", "native", "native", "stock"]
    rows = {
        mode: [row for run in report["runs"] if run["mode"] == mode for row in run["rows"]]
        for mode in ["stock", "native"]
    }
    lines = [
        "# Paired serving results",
        "",
        f"GPU: {report['gpu']}",
        f"Model: `{report['model']}` at `{report['revision']}`.",
        "",
        "ABBA process order on one GPU; three repeats per process. Latencies include",
        "local HTTP and client construction, but exclude fixture generation and JSON",
        "serialization before the HTTP timer. This is a bounded diagnostic, not D4/F3 certification.",
        "",
        "| Request | Cache | n / mode | Stock TTFT ms | Native TTFT ms | Change | Stock total ms | Native total ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for kind in ["text", "single", "multi", "repeat", "heavy", "concurrent"]:
        for state in ["cold", "warm"]:
            selected = {
                mode: [r for r in data if r["kind"] == kind and r["requested_state"] == state]
                for mode, data in rows.items()
            }
            if not selected["stock"]:
                continue
            ttft = {
                m: statistics.median(r["ttft_ms"] for r in data) for m, data in selected.items()
            }
            total = {
                m: statistics.median(r["latency_ms"] for r in data) for m, data in selected.items()
            }
            change = (ttft["native"] / ttft["stock"] - 1) * 100
            lines.append(
                f"| {kind} | {state} | {len(selected['stock'])} | {ttft['stock']:.2f} | {ttft['native']:.2f} | {change:+.1f}% | {total['stock']:.2f} | {total['native']:.2f} |"
            )
    lines += [
        "",
        "Negative change means lower median TTFT. Samples are small and requests",
        "within a process share caches and hardware; these are descriptive medians.",
        "",
        "| Mode | Image processing median / p95 ms | Health median / p95 ms | Peak summed process RSS GiB | Concurrent requests/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in rows:
        runs = [run for run in report["runs"] if run["mode"] == mode]
        events = []
        for run in runs:
            events += [
                event
                for event in map(
                    json.loads,
                    (root / f"{run['index']}-{mode}-audit.jsonl").read_text().splitlines(),
                )
                if event["time"] >= run["start_time"]
            ]
        timings = [
            event["ms"] for event in events if event["event"] == "processor" and event["grid"]
        ]
        health = [value for run in runs for value in run["health_latency_ms"]]
        durations = [r["batch_seconds"] for r in rows[mode] if r["kind"] == "concurrent"][::4]
        rate = 4 * len(durations) / sum(durations)
        peak = max(run["peak_process_tree_rss_bytes"] for run in runs) / 1024**3
        lines.append(
            f"| {mode} | {statistics.median(timings):.2f} / {percentile(timings, 0.95):.2f} | {statistics.median(health):.2f} / {percentile(health, 0.95):.2f} | {peak:.2f} | {rate:.2f} |"
        )
    lines += [
        "",
        "Processor timing includes one processing call, which may contain multiple images.",
        "Health-response latency is a responsiveness proxy, not direct event-loop lag.",
        "RSS sums processes and may double-count shared pages. Concurrent throughput",
        "includes the fixed 16-token generation budget, HTTP and queueing.",
        "",
        "## Correctness witnesses",
        "",
    ]
    baseline = report["runs"][0]["rows"]
    for run in report["runs"]:
        assert len(run["rows"]) == len(baseline)
        same_usage = sum(
            a["usage"] == b["usage"] for a, b in zip(baseline, run["rows"], strict=True)
        )
        same_text = sum(a["text"] == b["text"] for a, b in zip(baseline, run["rows"], strict=True))
        lines.append(
            f"- Run {run['index']} {run['mode']}: {same_usage}/{len(baseline)} equal usage records, {same_text}/{len(baseline)} equal decoded outputs versus run 0; audit {run['witness']}."
        )
    lines += [
        "",
        "Equal decoded outputs and usage are bounded behavioral witnesses, not a logits",
        "equivalence proof. Inspect differences, raw server logs and audit records before",
        "making an adoption recommendation.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(summarize(args.results))
