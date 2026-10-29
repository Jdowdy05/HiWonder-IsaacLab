# Cloud preparation validation, October 29, 2026

This is a prepared, tested deployment bundle, not a record of a paid deployment.

- Full project CPU regression: **499 passed, 3 skipped, 19 subtests passed**.
  The skips were the CUDA-only stream test and two Linux process-group tests on
  Windows. Those Linux group paths were separately exercised below.
- Linux-only cloud fixture/process checks on an isolated Jetson diagnostics
  directory: **102 passed, 1 catalog-import test deselected**. These are standard
  library/control-flow fixtures, not Isaac Sim or RTX 4090 deployment on Jetson.
- Actual local **Windows/RTX 5090** RGB PPO with SGS and the nine-action Eka arm:
  **two iterations, four optimizer updates, finite parameters, BF16 autocast**.
  Native 640x480/30 Hz configured capture, physics, solver and randomization are
  preserved. Configured sensor rate does not establish real-time frame rate.
  Peak owned-process RSS was about 8.45 GiB under the external 9 GiB RSS guard.
- CUDA GEMM utility: FP32, FP16 and BF16 numerical checks passed at 1024-square
  size on the same local 5090. No unlock was installed or applied, and those
  numbers are not 4090 cloud throughput or an unlock comparison.
- Fresh source-only installation matched **all 169 payload hashes**, the pinned
  official v3.0.0 commit and the offline uv lock check. SDK binaries were not
  reinstalled during this check; existing mismatched source is refused.
- Linux bootstrap syntax and Unix line endings, PowerShell upload syntax,
  credential/path exclusions, resource/cost math and precision-resume contracts
  are covered. Real CPU nine-action optimizer/checkpoint restore passed in
  both FP32 and BF16 modes. An initial full-scene GPU resume attempt was
  deferred by the host headroom guard. After headroom recovered, a bounded
  full-scene exact resume passed: iteration2->4, optimizer updates4->8,
  BF16 preserved and parameters finite.
- Independent review resolved three findings: nested cgroup memory reserves,
  edited upstream-source reuse and GB/GiB quote labeling. No blocking finding
  remained at closure. The public query allocates **200 GB**, not 200 GiB.

The original HiWonder project and canonical IsaacLab-30 checkout were preserved.
No motor interface, calibration or physical controller was changed by cloud
preparation. No paid instance, account deposit, provider message or GPU kernel
modification was performed.

On the actual rental, require the runtime preflight, native wrist-camera smoke,
checkpoint-resume smoke and matched FP32/BF16 pilot before a long run. Linux
RTX 4090 rendering/learning, the optional boot-time unlock, end-to-end speedup and
task success remain unverified until those host-specific checks run.
