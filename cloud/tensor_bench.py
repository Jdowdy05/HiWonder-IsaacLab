# SPDX-License-Identifier: BSD-3-Clause
"""Bounded CUDA GEMM measurements; never loads an unlock or modifies a driver."""

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="unverified-host-state")
    parser.add_argument("--size", type=int, choices=(512, 1024, 2048, 4096), default=4096)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if not 3 <= args.repeats <= 200 or args.output.exists():
        raise SystemExit("Use3–200 repeats and a new output path")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; no CPU result can establish RTX throughput")
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,driver_version,power.limit", "--format=csv,noheader"], text=True
    ).strip()
    torch.manual_seed(20261029)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    report = dict(
        label=args.label,
        hardware=hardware,
        torch=torch.__version__,
        size=args.size,
        qualification="cuBLAS GEMM timings; not peak issue rate, unlock verification or RL speedup",
        modes={},
    )
    for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
        if name == "bf16" and not torch.cuda.is_bf16_supported():
            report["modes"][name] = dict(status="unsupported")
            continue
        a = torch.randn(args.size, args.size, device="cuda", dtype=dtype) / math.sqrt(args.size)
        b = torch.randn_like(a) / math.sqrt(args.size)
        expected = a.float() @ b.float()
        output = torch.empty_like(a)
        for _ in range(5):
            torch.mm(a, b, out=output)
        torch.cuda.synchronize()
        times = []
        for _ in range(args.repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            torch.mm(a, b, out=output)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        error = float((output.float() - expected).norm() / expected.norm().clamp_min(1e-12))
        tolerance = 1e-5 if name == "fp32" else 0.005 if name == "fp16" else 0.03
        if not torch.isfinite(output).all() or not math.isfinite(error) or error > tolerance:
            raise RuntimeError(f"{name} failed numeric check: relative L2 error={error}")
        median = statistics.median(times)
        report["modes"][name] = dict(
            median_ms=median,
            tflops=2 * args.size**3 / (median / 1000) / 1e12,
            relative_l2_error=error,
            numeric_check_pass=True,
        )
        del a, b, output, expected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
