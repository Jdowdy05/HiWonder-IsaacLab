"""Evaluate saved ACT offline and produce evidence artifacts. No robot or simulator API."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sqlite3
import io

import numpy as np
from PIL import Image, ImageDraw
import torch

from experiment_data import json_write, sha256, validate_initialization
from model_io import load_policy_checked
from train_act import PublicEpisodes, amp, evaluate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    root=Path(args.out).resolve();run=Path(args.run).resolve()
    root.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
    torch.set_num_threads(4);torch.set_num_interop_threads(2)
    device=torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(0.18)
    data=PublicEpisodes(args.cache,30)
    checkpoint=run/'best'
    weight_hash=validate_initialization(checkpoint,data.root/'manifest.json')
    policy=load_policy_checked(checkpoint,device=device)
    rng=np.random.default_rng(20261030)
    # More representative phase coverage than the four anchors used to select checkpoints.
    data.val_anchors=[(i,int(f)) for i,e in enumerate(data.episodes) if e['split']=='val'
                      for f in sorted(rng.choice(e['frames'],min(16,e['frames']),replace=False).tolist())]
    normal=evaluate(policy,data,device,8)
    blank=evaluate(policy,data,device,8,image_mode='mean_image')
    json_write(root/'uniform-heldout-validation.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=weight_hash,
                anchor_seed=20261030,anchors=data.val_anchors,anchor_method='16 uniformly sampled distinct frames per held-out episode',
                normal=normal,mean_image=blank,
                caveat='Same held-out episodes as model selection. This is phase coverage, not an independent unseen-domain test.'))

    # Preview held-out public scenes and the same-frame observations fed to ACT.
    canvas=Image.new('RGB',(896,3*220),(245,247,250));draw=ImageDraw.Draw(canvas)
    forecasts=[]
    for source in range(len(data.manifest['sources'])):
        episode_index=next(i for i,e in enumerate(data.episodes) if e['source']==source and e['split']=='val')
        e=data.episodes[episode_index]
        draw.text((8,source*220+4),data.manifest['sources'][source]['id']+' / held-out episode '+str(e['id']),fill=(20,25,35))
        for col,frame in enumerate(np.linspace(0,e['frames']-1,4,dtype=int)):
            data.get(episode_index,int(frame))  # Opens its bounded, read-only SQLite connection.
            blob=data.connections[source].execute('SELECT jpeg FROM frames WHERE video=? AND idx=?',
                         (e['video'],int(e['image_indices'][frame]))).fetchone()[0]
            with Image.open(io.BytesIO(blob)) as image:
                canvas.paste(image.convert('RGB'),(col*224,source*220+27))
            draw.text((col*224+5,source*220+197),f'frame {frame} / {frame/30:.1f} s',fill=(30,35,45))
        frame=e['frames']//2
        batch=data.batch([(episode_index,frame)],device)
        inputs={k:v for k,v in batch.items() if k.startswith('observation.')}
        with torch.inference_mode(),amp(device):prediction=policy.predict_action_chunk(inputs).float()[0].cpu().numpy()
        pred=prediction*data.stats['action_std']+data.stats['action_mean']
        truth=batch['action'][0].cpu().numpy()*data.stats['action_std']+data.stats['action_mean']
        forecasts.append(dict(source=data.manifest['sources'][source]['id'],episode=e['id'],frame=frame,
                              predicted=pred.tolist(),target=truth.tolist(),current_state=e['state'][frame].tolist()))
    canvas.save(root/'public-scene-diversity.png')
    json_write(root/'heldout-forecast-examples.json',dict(joint_order=data.manifest['joint_order'],examples=forecasts))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12.8,3.8))
    log=[json.loads(line) for line in (run/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    validations=[r for r in log if r.get('event')=='validation']
    if validations:
        axes[0].plot([r['step'] for r in validations],[r['macro_normalized_l1'] for r in validations],color='#2563eb',marker='o')
    axes[0].set(xlabel='Warm-start optimizer steps',ylabel='Normalized action L1',title='ACT held-out error during training')
    axes[0].grid(alpha=.2)
    names=['Cube / box','Sweets','Foam object']
    x=np.arange(3);width=.36
    axes[1].bar(x-width/2,[r['arm_mae_source_units'] for r in normal['per_source']],width,label='ACT',color='#2563eb')
    axes[1].bar(x+width/2,[r['hold_arm_mae_source_units'] for r in normal['per_source']],width,label='Current-state hold',color='#94a3b8')
    axes[1].set(xticks=x,xticklabels=names,ylabel='Arm MAE (source angle units)',title='One-second future chunks')
    axes[1].legend(fontsize=8)
    axes[2].bar(x-width/2,[r['gripper_opening_fraction_mae']*100 for r in normal['per_source']],width,label='ACT',color='#2563eb')
    axes[2].bar(x+width/2,[r['hold_gripper_opening_fraction_mae']*100 for r in normal['per_source']],width,label='Current-state hold',color='#94a3b8')
    axes[2].set(xticks=x,xticklabels=names,ylabel='Opening error (% observed range)',title='Virtual gripper approximation')
    fig.suptitle('Offline ACT trial | 40 held-out episodes | source arm calibration and real-world success unvalidated',fontsize=11)
    fig.tight_layout();fig.savefig(root/'training-and-validation.png',dpi=160);plt.close(fig)
    result=dict(checkpoint=str(checkpoint),sha256=weight_hash,uniform_validation_l1=normal['macro_normalized_l1'],
                mean_image_validation_l1=blank['macro_normalized_l1'],source_results=normal['per_source'])
    json_write(root/'RESULTS.json',result)
    print(json.dumps(result),flush=True)
    data.close()


if __name__=='__main__':main()
