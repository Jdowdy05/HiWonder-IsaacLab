# SPDX-License-Identifier: BSD-3-Clause
"""Report measured PPO cost per transition; tensor peaks are not used as RL speed."""

import argparse
import json
import math
import statistics
from pathlib import Path


def summarize_samples(fps, collection, learning, hourly_usd, *, discard=2):
    if not math.isfinite(hourly_usd) or hourly_usd <= 0 or discard < 0:
        raise ValueError("Need a positive actual hourly quote and nonnegative warmup discard")
    samples = list(fps)[discard:]
    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("No finite, positive post-warmup end-to-end FPS measurements")
    median = statistics.median(samples)
    fractions = [
        b / (a + b)
        for a, b in zip(list(collection)[discard:], list(learning)[discard:])
        if math.isfinite(a) and math.isfinite(b) and a >= 0 and b >= 0 and a + b > 0
    ]
    fraction = statistics.median(fractions) if fractions else None
    return dict(
        median_transitions_per_second=median,
        measured_iterations=len(samples),
        hourly_usd=hourly_usd,
        usd_per_million_transitions=hourly_usd * 1_000_000 / (3600 * median),
        median_learning_fraction=fraction,
        ideal_total_speedup_if_learner_doubles=None if fraction is None else 1 / (1 - fraction / 2),
        scope=(
            "Warm steady-state collection+learner throughput; setup, shutdown, storage-only and transfer fees separate"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--hourly-usd", type=float, required=True)
    parser.add_argument("--discard", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(args.run), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])

    def values(name):
        return [x.value for x in accumulator.Scalars(name)] if name in tags else []

    result = summarize_samples(
        values("Perf/total_fps"),
        values("Perf/collection_time"),
        values("Perf/learning_time"),
        args.hourly_usd,
        discard=args.discard,
    )
    result["run"] = str(args.run.resolve())
    for name in ("result.json", "cloud-status.json"):
        path = args.run / name
        if path.is_file():
            result[name] = json.loads(path.read_text())
    status = result.get("cloud-status.json", {})
    if "elapsed_seconds" in status:
        result["training_process_wall_cost_usd"] = status["elapsed_seconds"] / 3600 * args.hourly_usd
    if "started_unix_time" in status:
        points = accumulator.Scalars("Perf/total_fps")
        result["seconds_to_first_logged_iteration"] = points[0].wall_time - status["started_unix_time"]
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
