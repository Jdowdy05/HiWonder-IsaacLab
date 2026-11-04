"""Launch and monitor only this experiment's process tree. No global process matching."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil

DEFAULT_WALL_SECONDS = 1800


def exceeded(*, rss_gib, free_gib, elapsed_s, seconds):
    if rss_gib > 6:
        return "rss_limit"
    if free_gib < 4:
        return "free_ram_reserve"
    if elapsed_s > seconds:
        return "wall_limit"
    return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',required=True)
    parser.add_argument('--seconds',type=int,default=DEFAULT_WALL_SECONDS)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command or args.seconds<1:
        parser.error('An explicit command and positive time limit are required')
    report=Path(args.report).resolve();report.parent.mkdir(parents=True,exist_ok=True)
    if report.exists():
        raise FileExistsError(report)
    log=report.with_suffix('.stdout.log')
    if log.exists():
        raise FileExistsError(log)
    start=time.monotonic();peak=0.;minimum=float('inf');reason=None;owned_pids=set()
    def write(status,code=None):
        value=dict(status=status,command=command,owned_pids=sorted(owned_pids),exit_code=code,
                   stop_reason=reason,elapsed_s=time.monotonic()-start,peak_rss_gib=peak,
                   min_free_ram_gib=minimum if minimum!=float('inf') else None)
        report.write_bytes((json.dumps(value,indent=2)+'\n').encode('utf-8'))
    with open(log,'w',encoding='utf-8',buffering=1) as output:
        child=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        owned=psutil.Process(child.pid);owned_pids.add(child.pid)
        write('running')
        try:
            while child.poll() is None:
                try:
                    processes=[owned]+owned.children(recursive=True)
                    rss=0
                    for process in processes:
                        try:
                            owned_pids.add(process.pid);rss+=process.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    rss=0
                peak=max(peak,rss/1024**3);minimum=min(minimum,psutil.virtual_memory().available/1024**3)
                reason=exceeded(rss_gib=rss/1024**3,free_gib=psutil.virtual_memory().available/1024**3,
                                elapsed_s=time.monotonic()-start,seconds=args.seconds)
                if reason:
                    # All victims are obtained through this exact spawned process's parent relationship.
                    for process in reversed(processes):
                        try:process.kill()
                        except psutil.NoSuchProcess:pass
                    child.wait(timeout=15)
                    break
                time.sleep(0.5)
        finally:
            if child.poll() is None:
                for process in reversed([owned]+owned.children(recursive=True)):
                    try:process.kill()
                    except psutil.NoSuchProcess:pass
                child.wait(timeout=15)
            write('stopped_by_guard' if reason else 'finished',child.returncode)
    print(report.read_text(encoding='utf-8'))
    raise SystemExit(child.returncode if child.returncode is not None else 1)


if __name__=='__main__':main()
