import argparse
import json
import math
import random
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from smind.model import checkpoint
from smind.model.checkpoint import SpatialCheckpoint
from smind.model.reader import SceneReader
from smind.model.readout import attach_token_points, load_maps
from smind.train.common import SOURCES, dataset, parse_with_config, scene_states

MAPS = "data/smind/fusion"


def readings(head, states, targets):
    totals, loss = {}, 0.0
    with torch.no_grad():
        for s, t in zip(states, targets):
            pred = head.read(head.belief(s), t)
            loss += float(head.loss(pred, t))
            for k, (a, b) in head.metrics(pred, t).items():
                x = totals.setdefault(k, [0, 0]); x[0] += a; x[1] += b
    return dict(loss=loss / len(states), **{k: a / b for k, (a, b) in totals.items()})


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="the checkpoint smind.train.fusion wrote (<out>/ckpt)")
    ap.add_argument("--out", required=True, help="the new checkpoint (same filesystem as --ckpt)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3, help="peak step of the readout (cosine to zero)")
    ap.add_argument("--layer_lr", type=float, default=0.05, help="peak step of the layer logits, as train.fusion")
    ap.add_argument("--maps", default=MAPS, help="the maps folder in the checkpoint's layout (smind.data.scene_group)")
    ap.add_argument("--seed", type=int, default=17)
    return parse_with_config(ap, argv)


def main():
    args = parse_args()
    ckpt = SpatialCheckpoint.load(args.ckpt)
    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(ckpt.path, local_files_only=True)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(ckpt.path, dtype=torch.bfloat16,
                                                        attn_implementation="sdpa", local_files_only=True).cuda().eval()
    reader = SceneReader(model, ckpt.token_ids(tokenizer), tokenizer.pad_token_id)
    states, maps = {}, {}
    for source, prefix in SOURCES.items():
        prefix = prefix.replace("data/smind/maps", args.maps, 1)
        for split in ("train", "dev", "dev_gray"):
            data, ms = dataset(f"{prefix}_{split}.parquet", tokenizer, processor), load_maps(f"{prefix}_{split}.jsonl")
            states[source, split], maps[source, split] = scene_states(reader, data), ms
    config, head = ckpt.readout()
    head = head.cuda()
    del model, reader
    torch.cuda.empty_cache()
    for ms in maps.values():
        attach_token_points(ms)
    targets = {k: [head.targets(m, "cuda") for m in ms] for k, ms in maps.items()}

    def evaluate():
        return {f"{src}_{split}": readings(head, states[src, split], targets[src, split])
                for src in SOURCES for split in ("dev", "dev_gray")}

    head.eval()
    before = evaluate()
    print("before", json.dumps(before), flush=True)
    train = [(src, i) for src in SOURCES for i in range(len(states[src, "train"]))]
    steps = args.epochs * math.ceil(len(train) / args.batch)
    rest = [p for n, p in head.named_parameters() if n != "layer_logits"]
    opt = torch.optim.AdamW([dict(params=[head.layer_logits], lr=args.layer_lr), dict(params=rest, lr=args.lr)],
                            weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: 0.5 * (1 + math.cos(math.pi * min(n, steps) / steps)))
    rng = random.Random(17)
    head.train()
    for epoch in range(args.epochs):
        rng.shuffle(train)
        total = 0.0
        for start in range(0, len(train), args.batch):
            group = train[start:start + args.batch]
            loss = sum(head.loss(head.read(head.belief(states[src, "train"][i]), targets[src, "train"][i]),
                                 targets[src, "train"][i]) for src, i in group) / len(group)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total += float(loss) * len(group)
        print(f"epoch {epoch + 1}/{args.epochs} train loss {total / len(train):.4f}", flush=True)
    head.eval()
    after = evaluate()
    print("after", json.dumps(after), flush=True)
    out = Path(args.out)
    checkpoint.link_model(ckpt.path, out)
    checkpoint.write_spatial(out, ckpt.layout, ckpt.insertion)
    checkpoint.save_readout(out, config, head)
    (out.parent / "refit.json").write_text(json.dumps(dict(args=vars(args), before=before, after=after), indent=1))


if __name__ == "__main__":
    main()
