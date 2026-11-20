import argparse
import datetime
import json
import math
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from smind.model import checkpoint
from smind.model.layout import RESERVED_TOKENS, Layout, register
from smind.model.reader import SceneReader, scene_fill
from smind.model.readout import attach_token_points, load_maps
from smind.train.common import LORA_TARGETS, dataset, evaluate, lr_multiplier, micro_batches, parse_with_config

MAPS = "data/smind/fusion"


def map_loss(reader, head, data, group, targets, fill, totals):
    _, input_ids, _ = reader.forward([data[i] for i in group], prefix_only=True, fill=fill)
    loss = 0.0
    for i, s in zip(group, reader.scene_states(input_ids)):
        pred = head.read(head.belief(s), targets[i])
        loss = loss + head.loss(pred, targets[i])
        for name, (a, b) in head.metrics(pred, targets[i]).items():
            x = totals.setdefault(name, [0, 0]); x[0] += a; x[1] += b
    reader.outputs = {}
    return loss


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the base model")
    ap.add_argument("--layout", required=True, help="m,c: spatial tokens per image, camera tokens (model.layout)")
    ap.add_argument("--train", nargs="+", default=[f"{MAPS}/maps_train", f"{MAPS}/maps_scannet_train"],
                    help="path prefixes: <p>.parquet (rows with spatial tokens) and <p>.jsonl (their maps, row for row); "
                         "every train image set once, not one row per SFT question, which would miss many sets and "
                         "repeat the common ones")
    ap.add_argument("--dev", nargs="+", default=[f"{MAPS}/maps_dev", f"{MAPS}/maps_scannet_dev",
                                                 f"{MAPS}/maps_scannetx_dev"],
                    help="dev map sets; each is also evaluated on its _gray copy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer_lr", type=float, default=0.05,
                    help="step of the learned layer logits (the same in smind.train.refit)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--embed_lr", type=float, default=3e-4,
                    help="spatial-token embeddings (Adam moves each weight about this much a step)")
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--local_points", action="store_true",
                    help="each cell's point in its own view's floor frame too, supervised on its own (VGGT's redundant "
                         "supervision; readout.ObservedGeometry local_points)")
    ap.add_argument("--local_conf", action="store_true",
                    help="with --local_points and --conf_alpha: the local points' own confidence "
                         "(readout.ObservedGeometry local_conf)")
    ap.add_argument("--conf_alpha", type=float, default=None,
                    help="a confidence per point with this alpha (none by default)")
    ap.add_argument("--lora_rank", type=int, default=0, help="0: frozen backbone; else LoRA of this rank, alpha 2x")
    ap.add_argument("--lora_lr", type=float, default=2e-4, help="peak step of the LoRA weights (warmup, cosine)")
    ap.add_argument("--micro_tokens", type=int, default=16384, help="padded tokens per micro-batch (checkpointed)")
    ap.add_argument("--eval_micro_tokens", type=int, default=4096, help="padded tokens per no-grad dev micro-batch")
    ap.add_argument("--evals", type=int, default=2, help="dev evaluations, evenly spaced, the last at the end")
    ap.add_argument("--seed", type=int, default=17)
    return parse_with_config(ap, argv)


def main():
    args = parse_args()
    layout = Layout.parse(args.layout)
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    if world > 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    lead = rank == 0
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                        attn_implementation="sdpa", local_files_only=True).cuda()
    model.requires_grad_(False)
    new_tokens = [t for t in layout.tokens + RESERVED_TOKENS if t not in tokenizer.get_vocab()]
    if len(tokenizer) + len(new_tokens) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer) + len(new_tokens), mean_resizing=False)
    embedding = model.get_input_embeddings().weight
    scene_ids, new = register(tokenizer, layout, embedding.shape[0])
    with torch.no_grad():
        base = embedding[:min(new)].float()
        mean, std, rms = base.mean(0), base.std(0), float(base.pow(2).mean().sqrt())
        del base
        embedding[new] = (mean + std * 0.1 * torch.randn(len(new), mean.shape[0], device="cuda")).to(embedding.dtype)
    scene = torch.nn.Parameter(embedding[scene_ids].detach().float().clone())
    fill = scene_fill(scene_ids, scene)

    data = torch.utils.data.ConcatDataset([dataset(p + ".parquet", tokenizer, processor) for p in args.train])
    maps = [m for p in args.train for m in load_maps(p + ".jsonl")]
    if len(data) != len(maps):
        raise ValueError(f"{len(data)} rows and {len(maps)} maps")
    devs = {Path(p).name + sfx: (dataset(p + sfx + ".parquet", tokenizer, processor), load_maps(p + sfx + ".jsonl"))
            for p in args.dev for sfx in ("", "_gray")}
    attach_token_points(maps)
    for _, ms in devs.values():
        attach_token_points(ms)
    per_epoch = math.ceil(len(data) / args.batch)
    steps = args.epochs * per_epoch
    lead and (out / "args.json").write_text(json.dumps(dict(vars(args), steps=steps, per_epoch=per_epoch, gpus=world,
                                                   trained="scene embeddings, readout" + (", LoRA" if args.lora_rank
                                                                                          else "")),
                                              indent=2))

    lora = []
    if args.lora_rank:
        from peft import LoraConfig, get_peft_model
        peft_model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.0,
                                                      target_modules=LORA_TARGETS))
        lora = [p for p in model.parameters() if p.requires_grad]
        if lead:
            print(f"LoRA: {sum(p.numel() for p in lora) / 1e6:.1f}M parameters in {len(lora)} tensors", flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    reader = SceneReader(model, scene_ids, tokenizer.pad_token_id)
    readout_args = {}
    if args.conf_alpha is not None:
        readout_args["conf_alpha"] = args.conf_alpha
    if args.local_points:
        readout_args["local_points"] = True
    if args.local_conf:
        readout_args["local_conf"] = True
    readout_args.update(m=layout.per_image, c=layout.camera)
    config = checkpoint.readout_config("geometry_group", len(reader.blocks), embedding.shape[1], readout_args,
                                       scene_ids)
    head = checkpoint.build_readout(config).cuda()
    map_targets = [head.targets(m, "cuda") for m in maps]
    dev_targets = {name: [head.targets(m, "cuda") for m in ms] for name, (_, ms) in devs.items()}
    rest = [p for n, p in head.named_parameters() if n != "layer_logits"]
    head_opt = torch.optim.AdamW([dict(params=[head.layer_logits], lr=args.layer_lr), dict(params=rest)],
                                 lr=args.head_lr, weight_decay=0.0)
    embed_opt = torch.optim.AdamW([scene], lr=args.embed_lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(embed_opt, lambda n: lr_multiplier(n, steps, int(steps * 0.03)))
    if lora:
        lora_opt = torch.optim.AdamW(lora, lr=args.lora_lr, weight_decay=0.0)
        lora_scheduler = torch.optim.lr_scheduler.LambdaLR(lora_opt,
                                                           lambda n: lr_multiplier(n, steps, int(steps * 0.03)))

    shared = [scene, *head.parameters(), *lora]
    if world > 1:
        for x in shared:
            torch.distributed.broadcast(x.data, 0)

    def run_eval(step):
        model.eval()
        row = dict(step=step, **{name: evaluate(reader, head, d, dev_targets[name], args.eval_micro_tokens, fill)
                                 for name, (d, _) in devs.items()})
        model.train()
        with (out / "dev_eval.jsonl").open("a") as h:
            h.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    order = [i for e in range(args.epochs)
             for i in torch.randperm(len(data), generator=torch.Generator().manual_seed(args.seed + e)).tolist()]
    evals = {round(steps * (k + 1) / args.evals) for k in range(args.evals)}
    started = time.time()
    for step in range(1, steps + 1):
        e, k = divmod(step - 1, per_epoch)
        members = order[e * len(data):(e + 1) * len(data)][k * args.batch:(k + 1) * args.batch]
        totals, map_sum = {}, 0.0
        for group in micro_batches(reader, data, members[rank::world], True, args.micro_tokens):
            loss = map_loss(reader, head, data, group, map_targets, fill, totals)
            (loss / len(members)).backward()
            map_sum += float(loss)
        if world > 1:
            for x in shared:
                if x.grad is None:
                    x.grad = torch.zeros_like(x)
                torch.distributed.all_reduce(x.grad)
            parts = [None] * world
            torch.distributed.all_gather_object(parts, (map_sum, totals))
            map_sum, totals = sum(p[0] for p in parts), {}
            for _, t in parts:
                for name, (a, b) in t.items():
                    x = totals.setdefault(name, [0, 0]); x[0] += a; x[1] += b
        embed_norm = float(torch.nn.utils.clip_grad_norm_([scene], 1.0))
        head_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0))
        extra = {}
        if lora:
            extra["lora_grad_norm"] = float(torch.nn.utils.clip_grad_norm_(lora, 1.0))
            lora_opt.step(); lora_scheduler.step(); lora_opt.zero_grad(set_to_none=True)
        embed_opt.step(); head_opt.step(); scheduler.step()
        embed_opt.zero_grad(set_to_none=True); head_opt.zero_grad(set_to_none=True)
        w = torch.softmax(head.layer_logits.detach().float(), 0)
        row = dict(step=step, map_loss=map_sum / len(members), embed_grad_norm=embed_norm, head_grad_norm=head_norm,
                   layer_peak=int(w.argmax()), layer_peak_weight=float(w.max()),
                   layer_effective_n=float(torch.exp(-(w * w.log()).sum())), **extra,
                   embed_lr=scheduler.get_last_lr()[0], scene_rms=float(scene.detach().pow(2).mean().sqrt()) / rms,
                   seconds=round(time.time() - started), peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1),
                   **{name: a / b for name, (a, b) in totals.items()})
        if lead:
            with (out / "metrics.jsonl").open("a") as h:
                h.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step in evals:
            if lead:
                run_eval(step)
            if world > 1:
                torch.distributed.barrier()

    if not lead:
        torch.distributed.destroy_process_group()
        return

    if lora:
        model = peft_model.merge_and_unload()
    with torch.no_grad():
        weight = model.get_input_embeddings().weight
        weight[scene_ids] = scene.to(weight.dtype)
    checkpoint.save(out / "ckpt", model, processor, layout, None, readout=(config, head))
    (out / "DONE").write_text(json.dumps(dict(steps=steps, seconds=round(time.time() - started))))
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
