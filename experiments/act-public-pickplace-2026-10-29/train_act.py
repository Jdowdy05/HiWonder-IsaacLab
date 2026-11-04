"""Bounded offline ACT trial, not a physical robot controller or PPO checkpoint."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import random
import sqlite3
import time

import numpy as np
import psutil
from PIL import Image
import torch

from experiment_data import action_chunk, json_write, sha256, validate_initialization, portable_cache_path


class PublicEpisodes:
    def __init__(self, cache, chunk=30):
        self.root = Path(cache).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.chunk = chunk
        self.episodes, self.connections = [], {}
        for source in self.manifest["sources"]:
            for row in source["episode_records"]:
                with np.load(portable_cache_path(self.root, row["npz"])) as arrays:
                    self.episodes.append(dict(**row, state=arrays["state"], action=arrays["action"], image_indices=arrays["image_indices"]))
        self.train_sources = [[i for i, e in enumerate(self.episodes) if e["split"] == "train" and e["source"] == s]
                              for s in range(len(self.manifest["sources"]))]
        self.val_anchors = [(i, int(f)) for i, e in enumerate(self.episodes) if e["split"] == "val"
                            for f in np.linspace(0, max(0, e["frames"]-self.chunk), 4, dtype=int)]
        self.stats = {k: np.asarray(v, np.float32) for k,v in self.manifest["normalization"].items()}
        self.image_mean = np.asarray(self.manifest["image_normalization"]["mean"], np.float32)[:,None,None]
        self.image_std = np.asarray(self.manifest["image_normalization"]["std"], np.float32)[:,None,None]

    def sample(self, rng):
        source = int(rng.integers(len(self.train_sources)))
        episode = int(rng.choice(self.train_sources[source]))
        frame = int(rng.integers(self.episodes[episode]["frames"]))
        return episode, frame

    def get(self, episode, frame):
        e = self.episodes[episode]
        source = e["source"]
        if source not in self.connections:
            db = self.root / str(source) / "images.sqlite"
            self.connections[source] = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
            self.connections[source].execute("PRAGMA cache_size=-2048")
        result = self.connections[source].execute("SELECT jpeg FROM frames WHERE video=? AND idx=?", (e["video"], int(e["image_indices"][frame]))).fetchone()
        if result is None:
            raise ValueError("Missing synchronized camera frame")
        with Image.open(io.BytesIO(result[0])) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2,0,1) / 255
        actions, pad = action_chunk(e["action"], frame, self.chunk)
        return {
            "observation.state": (e["state"][frame] - self.stats["state_mean"]) / self.stats["state_std"],
            "observation.images.scene": (pixels-self.image_mean)/self.image_std,
            "action": (actions-self.stats["action_mean"])/self.stats["action_std"],
            "action_is_pad": pad, "raw_state": e["state"][frame], "source": source,
        }

    def batch(self, anchors, device):
        items = [self.get(*a) for a in anchors]
        return {k:torch.as_tensor(np.stack([x[k] for x in items]), device=device) for k in items[0]}

    def close(self):
        for connection in self.connections.values():
            connection.close()


def build_policy(chunk=30, pretrained=True):
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    config = ACTConfig(
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(9,)),
                        "observation.images.scene": PolicyFeature(type=FeatureType.VISUAL, shape=(3,168,224))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(9,))},
        chunk_size=chunk, n_action_steps=1, device="cuda", push_to_hub=False,
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1" if pretrained else None,
        optimizer_lr=1e-4, optimizer_lr_backbone=1e-5,
    )
    return ACTPolicy(config)


def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()


@torch.inference_mode()
def evaluate(policy, data, device, batch_size, image_mode="normal"):
    policy.eval()
    sums = {i:dict(error=np.zeros(9), hold=np.zeros(9), normalized=0., count=0, first_error=np.zeros(9), first_count=0)
            for i in range(len(data.manifest["sources"]))}
    st = data.stats
    mean = torch.tensor(st["action_mean"],device=device)
    std = torch.tensor(st["action_std"],device=device)
    for start in range(0,len(data.val_anchors),batch_size):
        batch = data.batch(data.val_anchors[start:start+batch_size], device)
        inputs = {k:v for k,v in batch.items() if k.startswith("observation.")}
        if image_mode == "mean_image":
            inputs["observation.images.scene"] = torch.zeros_like(inputs["observation.images.scene"])
        with amp(device):
            prediction = policy.predict_action_chunk(inputs).float()
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite validation predictions")
        valid = ~batch["action_is_pad"]
        truth = batch["action"] * std + mean
        pred_raw = prediction * std + mean
        error = abs(pred_raw-truth).cpu().numpy()
        hold = abs(batch["raw_state"][:,None,:]-truth).cpu().numpy()
        normerr = abs(prediction-batch["action"]).cpu().numpy()
        valid_np = valid.cpu().numpy()
        for i, source in enumerate(batch["source"].cpu().tolist()):
            s = sums[source]; mask = valid_np[i]
            s["error"] += error[i,mask].sum(0); s["hold"] += hold[i,mask].sum(0)
            s["normalized"] += normerr[i,mask].sum(); s["count"] += int(mask.sum())
            s["first_error"] += error[i,0]; s["first_count"] += 1
    per_source = []
    for source, s in sums.items():
        err = s["error"]/s["count"]; hold=s["hold"]/s["count"]
        per_source.append(dict(source=data.manifest["sources"][source]["id"],
                               normalized_l1=float(s["normalized"]/s["count"]/9),
                               joint_mae_source_or_virtual_degrees=err.tolist(),
                               arm_mae_source_units=float(err[:5].mean()),
                               gripper_opening_fraction_mae=float(np.mean(err[[5,7]])/20),
                               hold_joint_mae=hold.tolist(), hold_arm_mae_source_units=float(hold[:5].mean()),
                               hold_gripper_opening_fraction_mae=float(np.mean(hold[[5,7]])/20),
                               first_action_joint_mae=(s["first_error"]/s["first_count"]).tolist(),
                               valid_targets=s["count"], anchors=s["first_count"]))
    return dict(image_mode=image_mode, macro_normalized_l1=float(np.mean([r["normalized_l1"] for r in per_source])), per_source=per_source)


def save_checkpoint(policy, path, data, step, metrics):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(path)
    json_write(path/"data_contract.json", data.manifest)
    json_write(path/"experiment_state.json", dict(step=step, validation=metrics, source_commit=data.manifest["upstream_lerobot_commit"],
               normalization="External: state/action train statistics; ImageNet RGB statistics. No normalization embedded in weights.",
               deployment_status=data.manifest["deployment_status"]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",required=True);parser.add_argument("--out",required=True)
    parser.add_argument("--steps",type=int,default=2000);parser.add_argument("--batch",type=int,default=8)
    parser.add_argument("--eval-every",type=int,default=250);parser.add_argument("--seed",type=int,default=20261029)
    parser.add_argument("--chunk",type=int,default=30);parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--initialize-from",type=Path,help="This experiment's matching ACT weights; starts a fresh optimizer and RNG")
    args=parser.parse_args()
    if args.steps<1 or not 1<=args.batch<=8 or args.chunk<1:
        raise ValueError("Invalid bounded trial configuration")
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]="1"
    torch.set_num_threads(4);torch.set_num_interop_threads(2)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda":
        raise RuntimeError("Actual ACT training trial requires CUDA")
    torch.cuda.set_per_process_memory_fraction(min(0.18,6*1024**3/torch.cuda.get_device_properties(0).total_memory))
    data=PublicEpisodes(args.cache,args.chunk)
    if args.smoke:
        data.val_anchors=data.val_anchors[::max(1,len(data.val_anchors)//8)][:8]
    policy=build_policy(args.chunk).to(device)
    initialization_sha=None
    if args.initialize_from is not None:
        initialization_sha=validate_initialization(args.initialize_from,data.root/'manifest.json')
        from safetensors.torch import load_model
        load_model(policy,str(args.initialize_from/'model.safetensors'))
    optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=1e-4,weight_decay=1e-4)
    rng=np.random.default_rng(args.seed)
    arguments=vars(args).copy()
    arguments['initialize_from']=str(args.initialize_from.resolve()) if args.initialize_from else None
    config=dict(**arguments,initialization_sha256=initialization_sha,device=str(device),gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                precision="BF16 autocast; FP32 parameters and optimizer",parameters=sum(p.numel() for p in policy.parameters()),
                dataset_manifest_sha256=sha256(data.root/"manifest.json"),source_balanced_sampling=True,
                train_episodes=sum(e["split"]=="train" for e in data.episodes),validation_episodes=sum(e["split"]=="val" for e in data.episodes),
                total_frames=sum(e["frames"] for e in data.episodes),image_augmentation="none; real source diversity only",
                training_job_pid=os.getpid())
    json_write(out/"run_config.json",config)
    started=time.monotonic(); initial=evaluate(policy,data,device,args.batch)
    json_write(out/"initial_validation.json",initial)
    best=float("inf");peak_rss=0;min_free=float("inf");steps_done=0
    with open(out/"metrics.jsonl","w",encoding="utf-8",buffering=1) as log:
        print(json.dumps(dict(event="training_start",**config,initial_l1=initial["macro_normalized_l1"])),flush=True)
        for step in range(1,args.steps+1):
            free=psutil.virtual_memory().available;rss=psutil.Process().memory_info().rss
            peak_rss=max(peak_rss,rss);min_free=min(min_free,free)
            if rss>6*1024**3 or free<4*1024**3 or time.monotonic()-started>1600:
                print(json.dumps(dict(event="graceful_resource_stop",step=step,rss=rss,free=free)),flush=True)
                break
            batch=data.batch([data.sample(rng) for _ in range(args.batch)],device)
            policy.train();optimizer.zero_grad(set_to_none=True)
            with amp(device):
                loss,loss_dict=policy(batch)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(policy.parameters(),10.)
            if not torch.isfinite(grad):
                raise ValueError("Nonfinite gradient")
            optimizer.step();steps_done=step
            if step%25==0 or step==1:
                row=dict(step=step,loss=float(loss.detach()),**loss_dict,gradient_norm=float(grad.detach()),elapsed_s=time.monotonic()-started,
                         rss_gib=rss/1024**3,cuda_allocated_gib=torch.cuda.memory_allocated()/1024**3)
                log.write(json.dumps(row)+"\n")
                if step%100==0 or step==1:print(json.dumps(row),flush=True)
            if step%args.eval_every==0 or step==args.steps:
                metrics=evaluate(policy,data,device,args.batch)
                row=dict(event="validation",step=step,**metrics);log.write(json.dumps(row)+"\n");print(json.dumps(row),flush=True)
                if metrics["macro_normalized_l1"]<best:
                    best=metrics["macro_normalized_l1"];save_checkpoint(policy,out/"best",data,step,metrics)
        final=evaluate(policy,data,device,args.batch)
        save_checkpoint(policy,out/"last",data,steps_done,final)
        torch.save(dict(optimizer=optimizer.state_dict(),step=steps_done,torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(),numpy_rng=rng.bit_generator.state),out/"last"/"optimizer_state.pt")
    # Reload the saved checkpoint and compare deterministic, prior-only action inference.
    from lerobot.policies.act.modeling_act import ACTPolicy
    test_batch=data.batch(data.val_anchors[:1],device)
    inputs={k:v for k,v in test_batch.items() if k.startswith("observation.")}
    policy.eval()
    with torch.inference_mode(),amp(device):reference=policy.predict_action_chunk(inputs).float().cpu()
    reloaded=ACTPolicy.from_pretrained(out/"last").to(device).eval()
    with torch.inference_mode(),amp(device):actual=reloaded.predict_action_chunk(inputs).float().cpu()
    reload_error=float((reference-actual).abs().max())
    if reload_error>1e-5:
        raise ValueError(f"Checkpoint reload changed inference: {reload_error}")
    del reloaded
    from safetensors.torch import load_model
    best_path=out/"best"
    if best_path.exists():load_model(policy,str(best_path/"model.safetensors"))
    best_metrics=evaluate(policy,data,device,args.batch)
    no_image=evaluate(policy,data,device,args.batch,image_mode="mean_image")
    summary=dict(steps=steps_done,initial_validation=initial,final_validation=final,best_validation=best_metrics,
                 mean_image_validation=no_image,checkpoint_reload_max_abs_error=reload_error,
                 elapsed_s=time.monotonic()-started,peak_sampled_rss_gib=peak_rss/1024**3,min_sampled_free_ram_gib=min_free/1024**3,
                 peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                 checkpoint_hashes={str(p.relative_to(out)):sha256(p) for p in out.glob("*/model.safetensors")},
                 status="OFFLINE_IMITATION_TRIAL_ONLY_NO_TASK_SUCCESS_OR_PHYSICAL_ACCEPTANCE")
    json_write(out/"summary.json",summary)
    print(json.dumps(dict(event="complete",steps=steps_done,best_l1=best_metrics["macro_normalized_l1"],reload_error=reload_error,elapsed_s=summary["elapsed_s"])),flush=True)
    data.close()


if __name__=="__main__":
    main()
