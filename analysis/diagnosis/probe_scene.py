import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from smind.model.checkpoint import SpatialCheckpoint, build_readout
from smind.model.reader import SceneReader
from smind.model.readout import attach_token_points, load_maps
from smind.train.common import SOURCES, dataset, scene_states

def fit(config, states, maps, epochs, seed, layer_logits):
    torch.manual_seed(seed)
    head = build_readout(config)
    del head.layer_logits
    head.register_buffer("layer_logits", layer_logits.clone())
    head = head.cuda()
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    targets = [head.targets(m, "cuda") for m in maps]
    order = list(range(len(states)))
    rng = random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), 64):
            loss = sum(head.loss(head.read(head.belief(states[i].cuda()), targets[i]), targets[i])
                       for i in order[start:start + 64])
            opt.zero_grad(); (loss / 64).backward(); opt.step()
    return head


def score(head, states, maps):
    per_set = []
    with torch.no_grad():
        for s, m in zip(states, maps):
            t = head.targets(m, "cuda")
            groups = [f"{m['task']}{m['n_views']}"] + ([f"room_{m['room']}"] if "room" in m else [])
            per_set.append((groups, head.metrics(head.read(head.belief(s.cuda()), t), t)))
    out, rng = {}, random.Random(0)
    for subset in ["all"] + sorted({g for gs, _ in per_set for g in gs}):
        sets = [r for gs, r in per_set if subset == "all" or subset in gs]
        for k in sorted({k for r in sets for k in r}):
            rows = [r[k] for r in sets if k in r]
            acc = sum(a for a, _ in rows) / sum(b for _, b in rows)
            boot = []
            for _ in range(2000):
                sample = [rows[rng.randrange(len(rows))] for _ in rows]
                boot.append(sum(a for a, _ in sample) / max(1, sum(b for _, b in sample)))
            boot.sort()
            out.setdefault(subset, {})[k] = dict(acc=acc, ci=[boot[49], boot[1949]], sets=len(rows), items=sum(b for _, b in rows))
    return out


def shuffled_maps(maps, seed):
    rng = random.Random(seed)
    groups = {}
    for i, m in enumerate(maps):
        groups.setdefault((m["source"], m["task"], m["n_views"]), []).append(i)
    out = list(maps)
    for idx in groups.values():
        perm = idx[:]; rng.shuffle(perm)
        for a, b in zip(idx, perm):
            out[a] = maps[b]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="a checkpoint with a readout")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30, help="epochs over the train sets for each fresh head")
    ap.add_argument("--scannet_train", type=int, default=2000, help="ScanNet train sets used (random, seed 17)")
    ap.add_argument("--maps", default="data/smind/probe",
                    help="the map folder in the checkpoint's layout (smind.data.scene_group)")
    args = ap.parse_args()
    ckpt = SpatialCheckpoint.load(args.ckpt)
    processor = AutoProcessor.from_pretrained(ckpt.path, local_files_only=True)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(ckpt.path, dtype=torch.bfloat16,
                                                        attn_implementation="sdpa", local_files_only=True).cuda().eval()
    config, saved = ckpt.readout()
    reader = SceneReader(model, ckpt.token_ids(tokenizer), tokenizer.pad_token_id)
    layer_logits = saved.layer_logits.detach().float()
    states, maps = {}, {}
    sources = {src: str(Path(args.maps) / Path(prefix).name) for src, prefix in SOURCES.items()}
    for source, prefix in sources.items():
        for split in ("train", "dev", "dev_gray"):
            data, ms = dataset(f"{prefix}_{split}.parquet", tokenizer, processor), load_maps(f"{prefix}_{split}.jsonl")
            keep = list(range(len(ms)))
            if source == "scannet" and split == "train":
                keep = sorted(random.Random(17).sample(keep, args.scannet_train))
            states[source, split] = scene_states(reader, torch.utils.data.Subset(data, keep))
            maps[source, split] = [ms[i] for i in keep]
    del model
    torch.cuda.empty_cache()
    train = [s for src in SOURCES for s in states[src, "train"]]
    train_maps = [m for src in SOURCES for m in maps[src, "train"]]
    zeros = lambda xs: [torch.zeros_like(x) for x in xs]
    rooms = {frozenset(o["name"] for o in m["objects"]) for m in maps["mindcube", "train"]}
    for split in ("dev", "dev_gray"):
        for m in maps["mindcube", split]:
            m["room"] = "seen" if frozenset(o["name"] for o in m["objects"]) in rooms else "new"
    for ms in maps.values():
        attach_token_points(ms)
    result = dict(ckpt=str(ckpt.path), readout=config["class"], train_sets={src: len(maps[src, "train"]) for src in SOURCES})
    true = fit(config, train, train_maps, args.epochs, 17, layer_logits)
    shuf = fit(config, train, shuffled_maps(train_maps, 17), args.epochs, 17, layer_logits)
    prior = fit(config, zeros(train), train_maps, args.epochs, 17, layer_logits)
    result["layer_weights"] = torch.softmax(true.layer_logits, 0).tolist()
    for src in SOURCES:
        dev, gray, dev_maps = states[src, "dev"], states[src, "dev_gray"], maps[src, "dev"]
        result[src] = dict(fresh_true=dict(dev=score(true, dev, dev_maps), gray=score(true, gray, dev_maps)),
                           fresh_shuffled=dict(dev=score(shuf, dev, dev_maps)),
                           fresh_zero_states=dict(dev=score(prior, zeros(dev), dev_maps)))
    Path(args.out).write_text(json.dumps(result, indent=2))
    for src in SOURCES:
        for arm, r in result[src].items():
            for split, subsets in r.items():
                for subset, metrics in subsets.items():
                    print(src, arm, split, subset, {m: round(x["acc"], 3) for m, x in metrics.items()})


if __name__ == "__main__":
    main()
