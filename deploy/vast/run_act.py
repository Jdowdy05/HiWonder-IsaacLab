"""Start one bounded ACT run in its own environment; no rental or RL processes are managed."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import uuid


def build_plan(workspace, source, *, steps=10000, smoke=False, from_scratch=False):
    if not isinstance(steps, int) or steps < 1:
        raise ValueError('Steps must be a positive integer')
    workspace, source = Path(workspace).resolve(), Path(source).resolve()
    label = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    experiment = source / 'experiments/act-public-pickplace-2026-10-29'
    python = workspace / '.venv/bin/python'
    output, report = workspace / 'runs' / label, workspace / 'jobs' / (label + '.json')
    command = [str(python), '-u', str(experiment / 'guard_trial.py'), '--report', str(report), '--seconds', '1800', '--',
               str(python), '-u', str(experiment / 'train_act.py'), '--cache', str(workspace / 'cache'),
               '--out', str(output), '--steps', str(steps), '--batch', '2' if smoke else '8',
               '--eval-every', str(steps if smoke else 1000)]
    if smoke:
        command.append('--smoke')
    if not from_scratch:
        command += ['--initialize-from', str(workspace / 'checkpoint')]
    return dict(command=command, output=str(output), report=str(report), from_scratch=from_scratch,
                semantics='Fresh optimizer/RNG; default initializes the archived best ACT weights with an identical data contract')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path('/workspace/act'))
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--steps', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--from-scratch', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    plan = build_plan(args.workspace, args.source, steps=args.steps if args.steps is not None else (3 if args.smoke else 10000),
                      smoke=args.smoke, from_scratch=args.from_scratch)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.dry_run:
        raise SystemExit(subprocess.run(plan['command'], check=False).returncode)


if __name__ == '__main__':
    main()
