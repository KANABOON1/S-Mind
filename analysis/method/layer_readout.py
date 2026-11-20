import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from smind.model.checkpoint import SpatialCheckpoint, build_readout
from smind.model.reader import SceneReader
from smind.model.readout import attach_token_points, load_maps
from smind.train.common import dataset, scene_states

MAPS = "data/smind/fusion"
PREFIX = {"mindcube": f"{MAPS}/maps", "scannet": f"{MAPS}/maps_scannet"}
METRICS = ["point_far", "point_far_v1", "point_far_vk", "local_far", "depth_ok", "cross_bearing", "facing", "view_hit"]


def read_states(ckpt, scannet_train):
    processor = AutoProcessor.from_pretrained(ckpt.path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(ckpt.path, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                        local_files_only=True).cuda().eval()
    reader = SceneReader(model, ckpt.token_ids(processor.tokenizer), processor.tokenizer.pad_token_id)
    out = {}
    for source, prefix in PREFIX.items():
        for split in ("train", "dev"):
            data = dataset(f"{prefix}_{split}.parquet", processor.tokenizer, processor)
            maps = load_maps(f"{prefix}_{split}.jsonl")
            keep = list(range(len(maps)))
            if source == "scannet" and split == "train":
                keep = sorted(random.Random(17).sample(keep, scannet_train))
            maps = attach_token_points([maps[i] for i in keep])
            out[source, split] = (scene_states(reader, torch.utils.data.Subset(data, keep)), maps)
            print(source, split, len(maps), flush=True)
    return out


def fit(config, states, maps, epochs, seed, layer_logits, device):
    torch.manual_seed(seed)
    head = build_readout(config)
    del head.layer_logits
    head.register_buffer("layer_logits", layer_logits.clone())
    head = head.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    targets = [head.targets(m, device) for m in maps]
    order = list(range(len(states)))
    rng = random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), 64):
            loss = sum(head.loss(head.read(head.belief(states[i].to(device)), targets[i]), targets[i])
                       for i in order[start:start + 64])
            opt.zero_grad(); (loss / 64).backward(); opt.step()
    return head


def score(head, states, maps, device):
    totals = {}
    with torch.no_grad():
        for s, m in zip(states, maps):
            t = head.targets(m, device)
            for k, (a, b) in head.metrics(head.read(head.belief(s.to(device)), t), t).items():
                x = totals.setdefault(k, [0, 0]); x[0] += a; x[1] += b
    return {k: round(a / b, 4) for k, (a, b) in totals.items() if k in METRICS and b}


def cache(ckpt, folder, scannet_train):
    data = read_states(ckpt, scannet_train)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({k: maps for k, (_, maps) in data.items()}, folder / "maps.pt")
    layers = next(iter(data.values()))[0][0].shape[0]
    for layer in range(layers):
        torch.save({k: [s[layer:layer + 1].clone() for s in states] for k, (states, _) in data.items()},
                   folder / f"layer_{layer}.pt")
    print("cached", layers, "layers", flush=True)


def probe(ckpt, folder, layers, epochs, out, device):
    config, saved = ckpt.readout()
    folder = Path(folder)
    maps = torch.load(folder / "maps.pt", weights_only=False)
    out = Path(out)
    result = json.loads(out.read_text()) if out.exists() else dict(ckpt=str(ckpt.path), epochs=epochs, layers={})
    for layer in layers:
        if layer == "zero":
            states = {k: [torch.zeros_like(s) for s in v] for k, v in torch.load(folder / "layer_0.pt").items()}
            logits = torch.zeros(1)
        elif layer == "mix":
            parts = [torch.load(folder / f"layer_{n}.pt") for n in range(saved.layer_logits.numel())]
            states = {k: [torch.cat([p[k][i] for p in parts]) for i in range(len(parts[0][k]))] for k in parts[0]}
            logits = saved.layer_logits.detach().float()
        else:
            states, logits = torch.load(folder / f"layer_{layer}.pt"), torch.zeros(1)
        train = [s for src in PREFIX for s in states[src, "train"]]
        train_maps = [m for src in PREFIX for m in maps[src, "train"]]
        head = fit(config, train, train_maps, epochs, 17, logits, device).eval()
        result["layers"][str(layer)] = {src: score(head, states[src, "dev"], maps[src, "dev"], device)
                                        for src in PREFIX}
        print(layer, result["layers"][str(layer)], flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["cache", "probe"])
    ap.add_argument("--ckpt", required=True, help="a fusion-stage checkpoint (train.refit's output)")
    ap.add_argument("--cache", required=True, help="the folder of per-layer states")
    ap.add_argument("--out", help="probe: the json written (layers already in it are kept)")
    ap.add_argument("--layers", nargs="*", default=[], help="probe: layers (from 0), mix and/or zero")
    ap.add_argument("--epochs", type=int, default=15, help="probe: epochs over the train sets for each fresh readout")
    ap.add_argument("--scannet_train", type=int, default=2000, help="cache: ScanNet train sets (random, seed 17)")
    ap.add_argument("--device", default="cuda", help="probe: where the fresh readouts are fit")
    args = ap.parse_args()
    ckpt = SpatialCheckpoint.load(args.ckpt)
    if args.step == "cache":
        cache(ckpt, args.cache, args.scannet_train)
    else:
        if not args.out:
            ap.error("probe needs --out")
        probe(ckpt, args.cache, [x if x in ("mix", "zero") else int(x) for x in args.layers], args.epochs, args.out,
              args.device)


if __name__ == "__main__":
    main()
