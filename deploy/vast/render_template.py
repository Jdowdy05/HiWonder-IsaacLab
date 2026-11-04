"""Render Vast UI field values and an integrity-checked immutable startup command."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def render(revision, script_sha):
    if not re.fullmatch('[0-9a-f]{40}', revision) or not re.fullmatch('[0-9a-f]{64}', script_sha):
        raise ValueError('Expected immutable commit and SHA256 values')
    values=json.loads(Path(__file__).with_name('template.json').read_text(encoding='utf-8'))
    url=f'https://raw.githubusercontent.com/Jdowdy05/HiWonder-IsaacLab/{revision}/deploy/vast/provision.sh'
    path=f'/workspace/hiwonder-template/provision-{revision}.sh'
    values['source_revision']=revision
    values['provision_sha256']=script_sha
    values['onstart']=(
        '#!/usr/bin/env bash\nset -Eeuo pipefail\nmkdir -p /workspace/hiwonder-template\n'
        f"curl -fsSL --retry 3 '{url}' -o '{path}'\n"
        f"printf '%s  %s\\n' '{script_sha}' '{path}' | sha256sum -c -\n"
        f"bash '{path}' '{revision}'\n"
    )
    return values


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--revision')
    parser.add_argument('--onstart-only',action='store_true')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]
    revision=args.revision or subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('Invalid revision')
    # Hash the selected committed file, not a possibly edited working-tree copy.
    source=subprocess.check_output(['git','-C',str(repo),'show',revision+':deploy/vast/provision.sh'])
    values=render(revision,hashlib.sha256(source).hexdigest())
    print(values['onstart'] if args.onstart_only else json.dumps(values,indent=2))


if __name__=='__main__':main()
