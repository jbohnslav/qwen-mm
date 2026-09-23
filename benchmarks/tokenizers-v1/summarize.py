"""Summarize the matched wheel runs and require exact before/after output parity."""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    before, after = [
        json.loads((ROOT / f"{label}.json").read_text()) for label in ("before", "after")
    ]
    before_pairs = {pair["pair_id"]: pair for pair in before["pairs"]}
    after_pairs = {pair["pair_id"]: pair for pair in after["pairs"]}
    if before_pairs.keys() != after_pairs.keys():
        raise ValueError("before/after case inventories differ")
    for key, pair in before_pairs.items():
        other = after_pairs[key]
        if (
            pair["implementations"]["candidate"]["output_signature"]
            != other["implementations"]["candidate"]["output_signature"]
        ):
            raise ValueError(f"before/after output mismatch: {key}")
        for item in (pair, other):
            check = item["implementations"]["candidate"]["conformance"]
            if (
                check["pre_measurement"] != "pass"
                or check["post_measurement"] != "pass"
                or not check["all_measured_iterations_stable"]
            ):
                raise ValueError(f"failed paired conformance: {key}")
    rows = []
    old = {(s["profile_alias"], s["case_id"]): s for s in before["summaries"]}
    for s in after["summaries"]:
        key = (s["profile_alias"], s["case_id"])
        old_ms, new_ms = old[key]["candidate"]["p50_ms"], s["candidate"]["p50_ms"]
        matching = [
            [p for p in data["pairs"] if (p["profile_alias"], p["case_id"]) == key]
            for data in (before, after)
        ]
        rss = [
            statistics.median(
                p["implementations"]["candidate"]["resource_census"]["rss"]["peak_rss_bytes"]
                for p in pairs
            )
            / 2**20
            for pairs in matching
        ]
        transient = [
            statistics.median(
                p["implementations"]["candidate"]["resource_census"]["rss"]["transient_rss_bytes"]
                for p in pairs
            )
            / 2**20
            for pairs in matching
        ]
        units = matching[0][0]["work_units"]
        rows.append(
            {
                "profile": key[0],
                "case": key[1],
                "before_ms": old_ms,
                "after_ms": new_ms,
                "speedup": old_ms / new_ms,
                "before_units_per_s": units * 1000 / old_ms,
                "after_units_per_s": units * 1000 / new_ms,
                "before_peak_rss_mib": rss[0],
                "after_peak_rss_mib": rss[1],
                "before_transient_rss_mib": transient[0],
                "after_transient_rss_mib": transient[1],
            }
        )
    (ROOT / "comparison.json").write_text(
        json.dumps(
            {"exact_output_parity": True, "pair_count": len(before_pairs), "rows": rows}, indent=2
        )
        + "\n"
    )
    lines = [
        "# Before/after results",
        "",
        "Local M4, release builds, one thread. Latencies are end-to-end preprocessing p50s, not pure tokenizer timings. Speedup is old qwen-mm latency divided by upgraded qwen-mm latency. See README.md for method and limits.",
        "",
        "| Profile | Case | Before ms | After ms | Speedup | Before units/s | After units/s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['profile']} | {r['case']} | {r['before_ms']:.3f} | {r['after_ms']:.3f} | {r['speedup']:.2f}x | {r['before_units_per_s']:.1f} | {r['after_units_per_s']:.1f} |"
        )
    lines += [
        "",
        "Units are prompts for text cases and images for image cases (24 images in 24 requests for jpeg24_requests). All 16 matched process/case pairs have byte-identical before/after output signatures, including token IDs, masks, and pixels. Both builds also passed paired oracle checks.",
        "",
        "## Memory",
        "",
        "Median process RSS census across two repetitions, in MiB. Includes reference/oracle allocations; this is not tokenizer-only memory or a hard bound.",
        "",
        "| Profile | Case | Peak before | Peak after | Transient before | Transient after |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['profile']} | {r['case']} | {r['before_peak_rss_mib']:.1f} | {r['after_peak_rss_mib']:.1f} | {r['before_transient_rss_mib']:.1f} | {r['after_transient_rss_mib']:.1f} |"
        )
    (ROOT / "comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
