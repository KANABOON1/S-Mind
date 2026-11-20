import argparse
import collections
import hashlib
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed"
IMG_OUT = OUT / "images"

PER_IMAGE_MAX_PIXELS = 640 * 480
PER_QUESTION_MAX_PIXELS = 4 * PER_IMAGE_MAX_PIXELS

OPTION_RE = re.compile(r"(?:^|\s)([A-H])\.\s")


def split_options(text):
    marks, want = [], "A"
    for m in OPTION_RE.finditer(text):
        if m.group(1) == want:
            marks.append(m)
            want = chr(ord(want) + 1)
    if len(marks) < 2:
        raise ValueError(f"cannot parse options: {text[:200]}")
    stem = text[: marks[0].start()].strip()
    opts = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        opts.append(f"{m.group(1)}. {text[m.end():end].strip()}")
    return stem, opts


FACTOR = 32
MIN_PIXELS = 65536
IMAGE_VERSION = "v2"


def resize_target(size, n_images):
    import math
    w, h = size
    cap = min(PER_IMAGE_MAX_PIXELS, PER_QUESTION_MAX_PIXELS // n_images)
    hb = max(FACTOR, round(h / FACTOR) * FACTOR)
    wb = max(FACTOR, round(w / FACTOR) * FACTOR)
    if hb * wb > cap:
        beta = math.sqrt((h * w) / cap)
        hb = max(FACTOR, math.floor(h / beta / FACTOR) * FACTOR)
        wb = max(FACTOR, math.floor(w / beta / FACTOR) * FACTOR)
    elif hb * wb < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (h * w))
        hb = math.ceil(h * beta / FACTOR) * FACTOR
        wb = math.ceil(w * beta / FACTOR) * FACTOR
    return wb, hb


def repo_path(path):
    return os.path.relpath(path, ROOT)


def materialize(src: Path, n_images: int) -> str:
    key = hashlib.md5(f"{repo_path(src)}|{n_images}|{IMAGE_VERSION}".encode()).hexdigest()
    dst = IMG_OUT / key[:2] / f"{key}{src.suffix.lower() or '.jpg'}"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(src)
        im = im.convert("RGB")
        tgt = resize_target(im.size, n_images)
        if tgt != im.size:
            im = im.resize(tgt, Image.BICUBIC)
        im.save(dst, quality=95)
    return str(dst)


def mindcube_group(qid):
    return qid.split("_q")[0]


def mindcube_setting(qid):
    prefix = qid.split("_")[0]
    return {"aroundnew": "around"}.get(prefix, prefix)


def _phash(path):
    import imagehash
    return int(str(imagehash.phash(Image.open(path).convert("RGB"))), 16)


def near_duplicate_pairs(paths, max_hamming):
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(64) as ex:
        arr = np.array(list(ex.map(_phash, paths, chunksize=64)), dtype=np.uint64)
    out = []
    for i in range(0, len(arr), 2048):
        x = arr[i:i + 2048, None] ^ arr[None, :]
        d = np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)
        for a, b in zip(*np.nonzero(d <= max_hamming)):
            if i + a < b:
                out.append((paths[i + a], paths[b]))
    return out


def build_mindcube():
    raw = RAW / "MindCube/data"
    scaffold = ROOT / "third_party/MindCube/data/scaffold/all/MindCube_train.jsonl"
    chains = {}
    for line in open(scaffold):
        x = json.loads(line)
        chains[x["id"]] = x["reasoning_chain"]
    recs = []
    for line in open(raw / "raw/MindCube_train.jsonl"):
        x = json.loads(line)
        stem, opts = split_options(x["question"])
        assert x["gt_answer"] in "ABCDEFGH"[: len(opts)], x["id"]
        recs.append(dict(
            source="mindcube", task=mindcube_setting(x["id"]), scene_id=mindcube_group(x["id"]),
            question_id=x["id"], raw_images=[str(raw / p) for p in x["images"]],
            question=stem, options=opts, answer=x["gt_answer"], reasoning_chain=chains[x["id"]],
        ))
    tiny = []
    for line in open(raw / "raw/MindCube_tinybench.jsonl"):
        x = json.loads(line)
        stem, opts = split_options(x["question"])
        tiny.append(dict(
            source="mindcube", task=mindcube_setting(x["id"]), scene_id=mindcube_group(x["id"]),
            question_id=x["id"], raw_images=[str(raw / p) for p in x["images"]],
            question=stem, options=opts, answer=x["gt_answer"], split="tiny",
        ))

    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in recs:
        for p in r["raw_images"]:
            parent[find("G:" + r["scene_id"])] = find("D:" + p.rsplit("/", 1)[0])
    for a, b in near_duplicate_pairs(sorted({p for r in recs for p in r["raw_images"]}), max_hamming=2):
        parent[find("D:" + a.rsplit("/", 1)[0])] = find("D:" + b.rsplit("/", 1)[0])
    for r in recs:
        r["group_id"] = r["scene_id"]
        root = find("G:" + r["group_id"])
        key = "D:" + repo_path(root[2:]) if root.startswith("D:") else root
        r["scene_id"] = "mc_comp_" + hashlib.md5(key.encode()).hexdigest()[:10]
    split = json.loads((ROOT / "configs/splits/mindcube.json").read_text())
    for r in recs:
        r["split"] = split[r["question_id"]]
    by_component = collections.defaultdict(set)
    for r in recs:
        by_component[r["scene_id"]].add(r["split"])
    if any(len(v) > 1 for v in by_component.values()):
        raise ValueError("configs/splits/mindcube.json puts one component's questions in two splits")
    return recs, tiny


def build_spatialladder(seed):
    raw = RAW / "SpatialLadder-26k"
    rows = [json.loads(l) for l in open(raw / "spld_spatial_data.jsonl")]
    subsets = {
        ("single_image", "relative direction"): "sl_single_direction",
        ("single_image", "relative distance"): "sl_single_distance",
        ("multi_view", "relative direction"): "sl_multiview_direction",
        ("multi_view", "relative distance"): "sl_multiview_distance",
    }
    pool = collections.defaultdict(list)
    for x in rows:
        task = subsets.get((x["data_type"], x["question_type"]))
        if task is None:
            continue
        imgs = x["image"] if isinstance(x["image"], list) else [x["image"]]
        scene = imgs[0].split("/")[0]
        opts = [o.strip() for o in x["options"]]
        assert all(o[0] == "ABCD"[i] and o[1] == "." for i, o in enumerate(opts)), opts
        opts = [f"{o[0]}. {o[2:].strip()}" for o in opts]
        pool[task].append(dict(
            source="spatialladder", task=task, scene_id=scene, question_id=f"spld_{x['question_id']}",
            raw_images=[str(raw / p) for p in imgs], question=x["question"].strip(), options=opts,
            answer=x["answer"].strip(),
        ))
    rng = random.Random(seed)

    room = lambda r: r["scene_id"].rsplit("_", 1)[0]
    rooms = sorted({room(r) for rs in pool.values() for r in rs})
    rng.shuffle(rooms)
    held, have = set(), collections.Counter()
    for s in rooms:
        if all(have[t] >= 50 for t in pool):
            break
        held.add(s)
        for t, rs in pool.items():
            have[t] += sum(room(r) == s for r in rs)
    dev = []
    for t, rs in pool.items():
        cand = [r for r in rs if room(r) in held]
        rng.shuffle(cand)
        for r in cand[:50]:
            dev.append(dict(r, split="dev"))

    def spread(rs, k):
        by = collections.defaultdict(list)
        for r in rs:
            by[r["scene_id"]].append(r)
        order = sorted(by)
        rng.shuffle(order)
        for s in order:
            rng.shuffle(by[s])
        out, rnd = [], 0
        while len(out) < k:
            took = False
            for s in order:
                if rnd < len(by[s]) and len(out) < k:
                    out.append(by[s][rnd])
                    took = True
            if not took:
                raise ValueError(f"only {len(out)} available, need {k}")
            rnd += 1
        return out

    def answer_capped(rs, k, max_frac):
        by = collections.defaultdict(list)
        for r in rs:
            by[r["answer"]].append(r)
        for v in by.values():
            rng.shuffle(v)
        cap = int(max_frac * k)
        take = {a: min(len(v), cap) for a, v in by.items()}
        while sum(take.values()) > k:
            a = max(take, key=take.get)
            take[a] -= 1
        assert sum(take.values()) == k, (take, k)
        return [r for a, v in by.items() for r in v[: take[a]]]

    train_pool = {t: [r for r in rs if room(r) not in held] for t, rs in pool.items()}
    train = []
    train += spread(train_pool["sl_single_direction"], 1300)
    train += answer_capped(train_pool["sl_multiview_direction"], 1300, 0.35)
    mv_dist = train_pool["sl_multiview_distance"]
    train += mv_dist
    train += spread(train_pool["sl_single_distance"], 4000 - len(train))
    assert len(train) == 4000, len(train)
    for r in train:
        r["split"] = "train"
    return train, dev, sorted(held)


def attach_images(recs, workers=32):
    def one(r):
        n = len(r["raw_images"])
        r["images"] = [materialize(Path(p), n) for p in r["raw_images"]]
        return r
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, recs))


def write(name, recs):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mc, tiny = build_mindcube()
    sl_train, sl_dev, sl_held = build_spatialladder(args.seed)
    all_recs = attach_images(mc + tiny + sl_train + sl_dev)

    split = lambda src, sp: [r for r in all_recs if r["source"] == src and r["split"] == sp]
    write("mindcube_train", split("mindcube", "train"))
    write("mindcube_dev", split("mindcube", "dev"))
    write("mindcube_tiny", split("mindcube", "tiny"))
    write("spatialladder_train", split("spatialladder", "train"))
    write("spatialladder_dev", split("spatialladder", "dev"))
    frames = collections.defaultdict(set)
    for r in split("spatialladder", "train") + split("spatialladder", "dev"):
        for p in r["raw_images"]:
            frames[Path(p).parent.name].add(int(Path(p).stem))
    (RAW / "ScanNet").mkdir(parents=True, exist_ok=True)
    (RAW / "ScanNet/sl_frames.json").write_text(json.dumps({s: sorted(v) for s, v in sorted(frames.items())}))

    stats = {
        "mindcube": {sp: collections.Counter(r["task"] for r in split("mindcube", sp)) for sp in ("train", "dev", "tiny")},
        "spatialladder": {sp: collections.Counter(r["task"] for r in split("spatialladder", sp)) for sp in ("train", "dev")},
        "spatialladder_heldout_scenes": sl_held,
        "seed": args.seed,
    }
    (OUT / "build_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in stats.items() if k != "spatialladder_heldout_scenes"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
