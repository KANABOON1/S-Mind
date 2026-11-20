import argparse
import datetime
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor

from smind.model import checkpoint
from smind.model.checkpoint import SpatialCheckpoint
from smind.model.insertion import inserted_mask
from smind.model.reader import SceneReader
from smind.model.readout import attach_token_points, load_maps
from smind.model.views import camera_poses, resolve, viewer_frame
from smind.train.common import (LORA_TARGETS, dataset, evaluate, lr_multiplier, micro_batches, parse_with_config,
                                 token_ce_sum)

DATA = "data/smind/sft/gpt-6-astra/perspective"
DEV = ["data/smind/fusion/maps_dev", "data/smind/fusion/maps_scannet_dev"]


class Terms:

    def __init__(self, head, layout):
        self.head, self.layout = head, layout

    def __call__(self, states, t, specs, objects, view):
        head, R = self.head, self.layout.run_length(view["n_views"])
        if states.shape[1] % R:
            raise ValueError(f"{states.shape[1]} scene states, not runs of {R}")
        z0 = head.belief(states[:, :R]).float()
        geo = head.loss(head.read(z0, t), t)
        read0 = head.read(z0, view)
        p0, cams0 = read0["points"].float().detach(), camera_poses(read0["camera"].float().detach())
        own = torch.stack([viewer_frame(p0, pose) for pose in cams0])
        mask = t["seen"] if "seen" in t else torch.ones(len(p0), dtype=torch.bool, device=p0.device)
        if len(mask) != len(p0):
            raise ValueError(f"{len(p0)} token cells, {len(mask)} in the map")
        objs = (torch.tensor([o["xy"] for o in objects], dtype=torch.float32, device=p0.device) / t["scale"]
                if objects and "scale" in t else None)
        poses = resolve(specs, cams0, objs)
        vloss, supervised, hits, cam_lines, cam = geo.new_zeros(()), 0, 0, 0, 0
        n = states.shape[1] // R - 1
        if n != len(specs):
            raise ValueError(f"{n} spatial-token runs in the answer, {len(specs)} view targets")
        for k, (spec, pose) in enumerate(zip(specs, poses)):
            zk = head.belief(states[:, (k + 1) * R:(k + 2) * R]).float()
            got = head.read(zk, view)["points"].float()[mask]
            if pose is not None:
                want = viewer_frame(p0, pose)[mask]
                err = F.smooth_l1_loss(got, want, beta=0.1, reduction="none").sum(-1)
                vloss = vloss + err.mean()
                supervised += 1
            cam = spec["image"] if spec["kind"] == "camera" else None
            if cam is not None:
                with torch.no_grad():
                    hits += int((got[None] - own[:, mask]).norm(dim=-1).mean(-1).argmin() == cam)
                cam_lines += 1
        return dict(geo=geo, view=vloss, supervised=supervised, hits=hits, cam_lines=cam_lines)


def forward_terms(reader, lm_head, items, token_ids, terms, extras):
    hid, input_ids, _ = reader.forward(items)
    labels = input_ids[:, 1:].clone()
    mask = torch.zeros_like(labels)
    for b, item in enumerate(items):
        mask[b, :len(item["loss_mask"]) - 1] = item["loss_mask"][1:]
    labels[~mask.bool() | inserted_mask(labels, input_ids[:, :-1], token_ids)] = -100
    with torch.autocast(hid.device.type, dtype=torch.bfloat16, enabled=hid.device.type == "cuda"):
        ce, tokens = token_ce_sum(hid[:, :-1].flatten(0, 1), labels.flatten(), lm_head)
    sums = dict(geo=0.0, view=0.0, supervised=0, hits=0, cam_lines=0)
    for b, item in enumerate(items):
        pos = torch.isin(input_ids[b, :len(item["input_ids"])], token_ids).nonzero()[:, 0]
        states = torch.stack([reader.outputs[j][b, pos] for j in range(len(reader.blocks))])
        grid = [(int(h) // 2, int(w) // 2) for _, h, w in item["multi_modal_inputs"]["image_grid_thw"].tolist()]
        view = dict(n_views=len(grid), grid=grid, view_index=torch.arange(1, len(grid), device=hid.device))
        r = terms(states, *extras[b], view)
        for key in r:
            sums[key] = sums[key] + r[key]
    reader.outputs = {}
    return ce, tokens, sums


def ce_tokens(item, token_ids):
    ids = item["input_ids"]
    return int((item["loss_mask"][1:].bool() & ~inserted_mask(ids[1:], ids[:-1], token_ids)).sum())


def splits(data_dir):
    return [s for s in ("train", "heldout") if (Path(data_dir) / f"{s}_z.parquet").exists()]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion_ckpt", required=True,
                    help="the fusion-stage checkpoint (train.refit's output): model, layout and readout")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--dev", nargs="+", default=DEV, help="the fusion stage's dev geometry sets, read off the prompt's spatial tokens at every evaluation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--geo_weight", type=float, default=1.0)
    ap.add_argument("--view_weight", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lora_rank", type=int, default=64)
    ap.add_argument("--lora_lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=float, default=0.1,
                    help="share of the steps the learning rate warms up over")
    ap.add_argument("--micro_tokens", type=int, default=16384, help="padded tokens per micro-batch (checkpointed)")
    ap.add_argument("--eval_micro_tokens", type=int, default=8192)
    ap.add_argument("--evals", type=int, default=4, help="evaluations after step 0, evenly spaced, the last at the end")
    ap.add_argument("--limit", type=int, help="the first rows of each split only (quick checks)")
    ap.add_argument("--seed", type=int, default=17)
    return parse_with_config(ap, argv)


def main():
    args = parse_args()
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    if world > 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    lead = rank == 0
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fusion_ckpt = SpatialCheckpoint.load(args.fusion_ckpt)
    start = fusion_ckpt.path

    processor = AutoProcessor.from_pretrained(start, local_files_only=True)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(start, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                        local_files_only=True).cuda()
    model.requires_grad_(False)
    layout, scene_ids = fusion_ckpt.layout, fusion_ckpt.token_ids(tokenizer)
    config, head = fusion_ckpt.readout()
    head = head.cuda().requires_grad_(False).eval()
    lm_head = model.lm_head
    token_ids = torch.tensor(scene_ids, dtype=torch.long, device="cuda")

    data_dir = Path(args.data)
    data, maps, specs, targets = {}, {}, {}, {}
    for s in splits(data_dir):
        data[s] = dataset(f"{data_dir}/{s}_z.parquet", tokenizer, processor)
        maps[s] = load_maps(f"{args.data}/{s}_maps.jsonl")
        specs[s] = json.loads(Path(f"{args.data}/{s}_targets.json").read_text())
        if not len(data[s]) == len(maps[s]) == len(specs[s]):
            raise ValueError(f"{s}: {len(data[s])} rows, {len(maps[s])} maps, {len(specs[s])} target lists")
        attach_token_points(maps[s])
        n = min(len(data[s]), args.limit or len(data[s]))
        targets[s] = [head.targets(m, "cuda") for m in maps[s][:n]]
    devs = {Path(p).name: (dataset(p + ".parquet", tokenizer, processor), attach_token_points(load_maps(p + ".jsonl")))
            for p in args.dev}
    dev_targets = {name: [head.targets(m, "cuda") for m in ms] for name, (_, ms) in devs.items()}
    n_train = min(len(data["train"]), args.limit or len(data["train"]))
    per_epoch = math.ceil(n_train / args.batch)
    steps = args.epochs * per_epoch
    lead and (out / "args.json").write_text(json.dumps(dict(vars(args), steps=steps, per_epoch=per_epoch, gpus=world,
                                                            layout=str(layout)), indent=2))

    from peft import LoraConfig, get_peft_model
    peft_model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.0,
                                                  target_modules=LORA_TARGETS))
    lora = [p for p in model.parameters() if p.requires_grad]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reader = SceneReader(model, scene_ids, tokenizer.pad_token_id)
    terms = Terms(head, layout)
    groups = [dict(params=lora, lr=args.lora_lr)]
    if world > 1:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        opt = ZeroRedundancyOptimizer(groups, optimizer_class=torch.optim.AdamW, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: lr_multiplier(n, steps, int(steps * args.warmup)))
    if lead:
        print(f"LoRA {sum(p.numel() for p in lora) / 1e6:.1f}M trained, {steps} steps, {world} GPU(s)", flush=True)

    def extras(s, group):
        return [(targets[s][i], specs[s][i], maps[s][i].get("objects")) for i in group]

    def dev_eval(step):
        model.eval()
        held = data.get("heldout", [])
        rows = range(min(len(held), args.limit or len(held)))
        tot = dict(ce=0.0, tokens=0, geo=0.0, view=0.0, supervised=0, hits=0, cam_lines=0)
        for group in micro_batches(reader, held, rows, False, args.eval_micro_tokens) if len(rows) else []:
            with torch.no_grad():
                ce, tokens, s = forward_terms(reader, lm_head, [held[i] for i in group], token_ids,
                                              terms, extras("heldout", group))
            tot["ce"] += float(ce); tot["tokens"] += tokens
            for key in ("geo", "view"):
                tot[key] += float(s[key])
            for key in ("supervised", "hits", "cam_lines"):
                tot[key] += s[key]
        row = dict(step=step)
        if len(rows):
            row["ce"] = tot["ce"] / tot["tokens"]
            row.update(geo=tot["geo"] / len(rows), view=tot["view"] / max(tot["supervised"], 1),
                       hit=tot["hits"] / max(tot["cam_lines"], 1))
        row.update(dev={name: evaluate(reader, head, d, dev_targets[name], args.eval_micro_tokens)
                        for name, (d, _) in devs.items()})
        model.train()
        with (out / "dev_eval.jsonl").open("a") as h:
            h.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    if lead:
        dev_eval(0)
    if world > 1:
        torch.distributed.barrier()
    train = data["train"]
    cpu_ids = token_ids.cpu()
    order = [i for e in range(args.epochs)
             for i in torch.randperm(n_train, generator=torch.Generator().manual_seed(args.seed + e)).tolist()]
    evals = {round(steps * (k + 1) / args.evals) for k in range(args.evals)}
    started = time.time()
    model.train()
    for step in range(1, steps + 1):
        e, k = divmod(step - 1, per_epoch)
        members = order[e * n_train:(e + 1) * n_train][k * args.batch:(k + 1) * args.batch]
        batch_tokens = sum(ce_tokens(train[i], cpu_ids) for i in members)
        batch_lines = sum(len(specs["train"][i]) for i in members)
        tot = dict(ce=0.0, geo=0.0, view=0.0, supervised=0, hits=0, cam_lines=0)
        for group in micro_batches(reader, train, members[rank::world], False, args.micro_tokens):
            ce, _, s = forward_terms(reader, lm_head, [train[i] for i in group], token_ids, terms,
                                     extras("train", group))
            loss = (ce / batch_tokens + args.geo_weight * s["geo"] / len(members)
                    + args.view_weight * s["view"] / batch_lines)
            loss.backward()
            tot["ce"] += float(ce.detach())
            for key in ("geo", "view"):
                tot[key] += float(torch.as_tensor(s[key]).detach())
            for key in ("supervised", "hits", "cam_lines"):
                tot[key] += s[key]
        if world > 1:
            for x in lora:
                if x.grad is None:
                    x.grad = torch.zeros_like(x)
                torch.distributed.all_reduce(x.grad)
            parts = [None] * world
            torch.distributed.all_gather_object(parts, tot)
            tot = {key: sum(part[key] for part in parts) for key in tot}
        norm = float(torch.nn.utils.clip_grad_norm_(lora, 1.0, error_if_nonfinite=True))
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        row = dict(step=step, ce=tot["ce"] / batch_tokens, geo=tot["geo"] / len(members),
                   view=tot["view"] / max(tot["supervised"], 1), hit=tot["hits"] / max(tot["cam_lines"], 1))
        row.update(grad_norm=norm, lr=sched.get_last_lr()[0], seconds=round(time.time() - started),
                   peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
        if lead:
            with (out / "metrics.jsonl").open("a") as h:
                h.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if step in evals:
                dev_eval(step)
        if world > 1:
            torch.distributed.barrier()

    if not lead:
        torch.distributed.destroy_process_group()
        return
    merged = peft_model.merge_and_unload()
    checkpoint.save(out / "ckpt", merged, processor, layout, "self", readout=(config, head))
    (out / "DONE").write_text(json.dumps(dict(steps=steps, seconds=round(time.time() - started))))
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
