"""Pinned public-data experiment. Raw data and source coordinate frames are preserved."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import sqlite3
import time
import urllib.request
from pathlib import Path, PurePosixPath

import numpy as np

SOURCES = [
    dict(id="lerobot/svla_so100_pickplace", revision="728583b5eaf9e739a7f119e2def466fa1d552402", camera="observation.images.top", open_is_high=False),
    dict(id="jmrog/so100_sweet_pick", revision="54141acb0bd6bcb868e34b4eb328f26481d333b2", camera="observation.images.laptop", open_is_high=False),
    dict(id="nbaron99/so100_pick_and_place4", revision="093be9af5954011b77e77bcd5f15b07ebdfcd980", camera="observation.images.laptop", open_is_high=True),
]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "left_finger_root", "left_finger_tip", "right_finger_root", "right_finger_tip"]


def portable_cache_path(root, relative):
    """Resolve saved Windows or POSIX cache names without changing the data contract."""
    if not isinstance(relative, str) or not relative:
        raise ValueError('Missing relative cache path')
    path = PurePosixPath(relative.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or any(':' in part for part in path.parts) or not path.parts:
        raise ValueError('Cache path must stay inside its root')
    return root.joinpath(*path.parts)


def canonical_six(raw, names):
    names = [n.removeprefix("main_").removesuffix(".pos") for n in names]
    expected = JOINTS[:5] + ["gripper"]
    if sorted(names) != sorted(expected):
        raise ValueError("Unsupported or duplicate source joint names")
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 6 or not np.isfinite(raw).all():
        raise ValueError("Expected finite six-joint source data")
    return raw[:, [names.index(n) for n in expected]]


def fit_normalization(episodes):
    train = [e for e in episodes if e["split"] == "train"]
    if not train:
        raise ValueError("No training episodes")
    stats = {}
    for key in ("state", "action"):
        samples = np.concatenate([e[key] for e in train]).astype(np.float64)
        stats[key + "_mean"] = samples.mean(axis=0).tolist()
        stats[key + "_std"] = np.maximum(samples.std(axis=0), 1e-3).tolist()
    return stats


def selected_paths(files: list[str], camera: str) -> list[str]:
    return [f for f in files if f == "README.md" or f.startswith("meta/")
            or (f.startswith("data/") and f.endswith(".parquet"))
            or (f.startswith("videos/") and camera in Path(f).parts and f.endswith(".mp4"))]


def split_episodes(ids, seed=20261029, fraction=0.2):
    ids = sorted(set(int(i) for i in ids))
    if len(ids) < 2:
        raise ValueError("At least two complete episodes are needed")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ids).tolist()
    n = min(len(ids) - 1, max(1, round(len(ids) * fraction)))
    return sorted(shuffled[n:]), sorted(shuffled[:n])


def action_chunk(actions, index, length):
    if not 0 <= index < len(actions) or length < 1:
        raise ValueError("Invalid chunk index/length")
    indices = np.arange(index, index + length)
    return actions[np.minimum(indices, len(actions) - 1)].copy(), indices >= len(actions)


def retarget_virtual(raw, *, closed, opened, open_is_high):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 6 or not np.isfinite(raw).all():
        raise ValueError("Expected finite six-joint samples")
    if not np.isfinite([closed, opened]).all() or abs(opened - closed) < 1e-4:
        raise ValueError("Degenerate gripper reference range")
    if (opened > closed) != open_is_high:
        raise ValueError("Gripper reference range disagrees with inspected direction")
    g = np.clip((raw[:, 5] - closed) / (opened - closed), 0, 1)
    hand = np.stack((-20 * g, 18 * (1 - g), -20 * g, 18 * (1 - g)), axis=1)
    return np.concatenate((raw[:, :5], hand), axis=1).astype(np.float32)


def matching_frame_indices(pts, timestamps, *, video_start, fps):
    pts = np.asarray(pts, dtype=np.float64)
    target = np.asarray(timestamps, dtype=np.float64) + video_start
    if not len(pts) or np.any(np.diff(pts) <= 0) or not np.isfinite(target).all():
        raise ValueError("Invalid video timestamps")
    right = np.minimum(np.searchsorted(pts, target), len(pts) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(abs(pts[left] - target) < abs(pts[right] - target), left, right)
    if np.max(abs(pts[indices] - target)) > 0.55 / fps:
        raise ValueError("Video and trajectory timestamps do not align within half a frame")
    return indices


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode("utf-8"))


def public_json(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return json.load(response)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_initialization(checkpoint, manifest):
    checkpoint = Path(checkpoint)
    contract = checkpoint / 'data_contract.json'
    if not contract.is_file() or sha256(contract) != sha256(manifest):
        raise ValueError('Initialization data contract differs from the current dataset/normalization')
    weights = checkpoint / 'model.safetensors'
    if not weights.is_file():
        raise ValueError('Initialization weights are missing')
    return sha256(weights)


def download(root):
    """Anonymous, pinned downloads; no credential discovery and no remote writes."""
    root = Path(root).resolve()
    for spec in SOURCES:
        dest = root / spec["id"].replace("/", "--")
        dest.mkdir(parents=True, exist_ok=True)
        api = public_json(f'https://huggingface.co/api/datasets/{spec["id"]}/revision/{spec["revision"]}?blobs=true')
        if api["sha"] != spec["revision"] or api.get("cardData", {}).get("license") != "apache-2.0":
            raise ValueError("Source revision or license changed")
        files = {f["rfilename"]: f for f in api["siblings"]}
        paths = selected_paths(list(files), spec["camera"])
        total = sum(files[p].get("size", 0) for p in paths)
        if total > 2_500_000_000:
            raise ValueError("Unexpected dataset download size")
        print(json.dumps(dict(event="download_start", source=spec["id"], files=len(paths), bytes=total)), flush=True)
        def fetch(name):
            path = dest / name
            expected = files[name].get("size")
            remote_sha = files[name].get("lfs", {}).get("sha256") or files[name].get("lfs", {}).get("oid")
            if not path.exists() or (expected is not None and path.stat().st_size != expected):
                path.parent.mkdir(parents=True, exist_ok=True)
                for attempt in range(3):
                    try:
                        url = f'https://huggingface.co/datasets/{spec["id"]}/resolve/{spec["revision"]}/{name}'
                        with urllib.request.urlopen(url, timeout=60) as response, open(str(path) + ".part", "wb") as out:
                            for block in iter(lambda: response.read(1024 * 1024), b""):
                                out.write(block)
                        if expected is not None and Path(str(path) + ".part").stat().st_size != expected:
                            raise ValueError("Incomplete download")
                        Path(str(path) + ".part").replace(path)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(1 + attempt)
            digest = sha256(path)
            if remote_sha and len(remote_sha) == 64 and digest != remote_sha:
                raise ValueError(f"Remote hash mismatch: {name}")
            return dict(path=name, bytes=path.stat().st_size, sha256=digest, remote_sha256=remote_sha)
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for result in pool.map(fetch, paths):
                results.append(result)
                if len(results) % 25 == 0:
                    print(json.dumps(dict(event="download_progress", source=spec["id"], files=len(results))), flush=True)
        json_write(dest / "download_manifest.json", dict(**spec, license="apache-2.0", files=results))
        print(json.dumps(dict(event="download_complete", source=spec["id"], files=len(results), bytes=sum(r["bytes"] for r in results))), flush=True)


def read_episodes(root, spec):
    import pyarrow.parquet as pq

    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if info["fps"] != 30:
        raise ValueError("This experiment expects a common 30 Hz control rate")
    names = {k: info["features"][k]["names"] for k in ("action", "observation.state")}
    metadata = {}
    if info["codebase_version"].startswith("v3"):
        for path in sorted(root.glob("meta/episodes/**/*.parquet")):
            for row in pq.read_table(path).to_pylist():
                metadata[int(row["episode_index"])] = row
    episodes = []
    for path in sorted(root.glob("data/**/*.parquet")):
        table = pq.read_table(path).to_pydict()
        episode_ids = np.asarray(table["episode_index"], dtype=np.int64)
        for episode_id in np.unique(episode_ids):
            mask = episode_ids == episode_id
            frame_indices = np.asarray(table["frame_index"])[mask]
            order = np.argsort(frame_indices)
            if not np.array_equal(frame_indices[order], np.arange(mask.sum())):
                raise ValueError("Episode frames are missing, duplicated or split across files")
            timestamp = np.asarray(table["timestamp"], np.float64)[mask][order]
            if not np.allclose(np.diff(timestamp), 1 / info["fps"], atol=0.001):
                raise ValueError("Nonuniform trajectory frame timing")
            if metadata:
                row = metadata[int(episode_id)]
                prefix = "videos/" + spec["camera"]
                video = info["video_path"].format(video_key=spec["camera"], chunk_index=int(row[prefix + "/chunk_index"]), file_index=int(row[prefix + "/file_index"]))
                video_start = float(row[prefix + "/from_timestamp"])
                if int(row["length"]) != len(order):
                    raise ValueError("Episode metadata length mismatch")
            else:
                video = info["video_path"].format(video_key=spec["camera"], episode_chunk=int(episode_id) // info["chunks_size"], episode_index=int(episode_id))
                video_start = 0.0
            episodes.append(dict(id=int(episode_id), timestamp=timestamp,
                                 state=canonical_six(np.asarray(table["observation.state"])[mask][order], names["observation.state"]),
                                 action=canonical_six(np.asarray(table["action"])[mask][order], names["action"]),
                                 video=video, video_start=video_start))
    if len(episodes) != info["total_episodes"] or sum(len(e["state"]) for e in episodes) != info["total_frames"]:
        raise ValueError("Downloaded trajectories do not match source episode/frame totals")
    return episodes, info


def cache_video(db, path, video):
    import av
    from PIL import Image

    done = db.execute("SELECT count FROM videos WHERE video=?", (video,)).fetchone()
    if done:
        return np.array([r[0] for r in db.execute("SELECT pts FROM frames WHERE video=? ORDER BY idx", (video,))])
    times = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 2
        for idx, frame in enumerate(container.decode(video=0)):
            pts = float(frame.pts * frame.time_base)
            im = frame.to_image().resize((224, 168), Image.Resampling.BILINEAR)
            jpeg = io.BytesIO()
            im.save(jpeg, format="JPEG", quality=90)
            db.execute("INSERT OR REPLACE INTO frames VALUES(?,?,?,?)", (video, idx, pts, jpeg.getvalue()))
            times.append(pts)
            if idx % 500 == 0:
                db.commit()
    db.execute("INSERT OR REPLACE INTO videos VALUES(?,?)", (video, len(times)))
    db.commit()
    return np.asarray(times)


def prepare(root, cache):
    root, cache = Path(root).resolve(), Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    all_episodes, public_manifest = [], []
    for source_index, spec in enumerate(SOURCES):
        source_root = root / spec["id"].replace("/", "--")
        source_cache = cache / str(source_index)
        source_cache.mkdir(parents=True, exist_ok=True)
        episodes, info = read_episodes(source_root, spec)
        train_ids, val_ids = split_episodes([e["id"] for e in episodes])
        # Endpoints are observed training ranges, not physical full-open/full-close calibration.
        train_grip = np.concatenate([e[k][:, 5] for e in episodes if e["id"] in train_ids for k in ("state", "action")])
        lo, hi = np.quantile(train_grip, [0.005, 0.995]).tolist()
        opened, closed = (hi, lo) if spec["open_is_high"] else (lo, hi)
        db = sqlite3.connect(source_cache / "images.sqlite")
        db.execute("PRAGMA cache_size=-4096")
        db.execute("CREATE TABLE IF NOT EXISTS frames(video TEXT, idx INTEGER, pts REAL, jpeg BLOB, PRIMARY KEY(video,idx))")
        db.execute("CREATE TABLE IF NOT EXISTS videos(video TEXT PRIMARY KEY,count INTEGER)")
        pts_by_video = {}
        meta_episodes = []
        for n, e in enumerate(episodes):
            video = e["video"]
            if video not in pts_by_video:
                pts_by_video[video] = cache_video(db, source_root / video, video)
            image_indices = matching_frame_indices(pts_by_video[video], e["timestamp"], video_start=e["video_start"], fps=info["fps"])
            split = "train" if e["id"] in train_ids else "val"
            q = retarget_virtual(e["state"], opened=opened, closed=closed, open_is_high=spec["open_is_high"])
            action = retarget_virtual(e["action"], opened=opened, closed=closed, open_is_high=spec["open_is_high"])
            npz = source_cache / f'episode-{e["id"]:06d}.npz'
            np.savez_compressed(npz, state=q, action=action, image_indices=image_indices, timestamp=e["timestamp"])
            item = dict(source=source_index, id=e["id"], split=split, frames=len(q), video=video, npz=str(npz.relative_to(cache)))
            meta_episodes.append(item)
            all_episodes.append(dict(**item, state=q, action=action))
            if n % 10 == 0:
                print(json.dumps(dict(event="prepare_progress", source=spec["id"], episodes=n+1, total=len(episodes))), flush=True)
        db.close()
        public_manifest.append(dict(**spec, frames=info["total_frames"], episodes=len(episodes), fps=30, image_size=[224,168],
                                    train_episode_ids=train_ids, validation_episode_ids=val_ids, gripper_open_reference=opened,
                                    gripper_closed_reference=closed, gripper_reference_method="train-only 0.5/99.5 percentiles; direction inspected in original video",
                                    arm_coordinates="source-native legacy angle coordinates; cross-source physical zero alignment unavailable",
                                    episode_records=meta_episodes, raw_manifest=str(source_root / "download_manifest.json")))
    normalization = fit_normalization(all_episodes)
    manifest = dict(schema=1, upstream_lerobot_commit="7e241bd630a3719a56157a497ce5d08f244784f1", sources=public_manifest,
                    fps=30, joint_order=JOINTS, normalization=normalization,
                    image_normalization=dict(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                    deployment_status="EXPERIMENTAL_OFFLINE_ONLY: source arm calibration, Eka aperture/contact geometry, and wrist camera transfer unvalidated")
    json_write(cache / "manifest.json", manifest)
    print(json.dumps(dict(event="prepared", episodes=len(all_episodes), frames=sum(e["frames"] for e in all_episodes))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["download", "prepare"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--cache")
    args = parser.parse_args()
    if args.operation == "download":
        download(args.root)
    else:
        prepare(args.root, args.cache)
