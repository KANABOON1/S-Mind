import argparse
import math

import torch
import yaml
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

LORA_TARGETS = (r".*language_model\.layers\.\d+\.(linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)"
                r"|self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))")


def parse_with_config(ap, argv=None):
    ap.add_argument("--config", help="YAML file of option values; options on the command line override it")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    path = pre.parse_known_args(argv)[0].config
    if path:
        values = yaml.safe_load(open(path)) or {}
        options = {a.dest: a for a in ap._actions}
        unknown = sorted(set(values) - set(options))
        if unknown:
            ap.error(f"{path}: not options of this command: {', '.join(unknown)}")
        for key in values:
            options[key].required = False
        ap.set_defaults(**values)
    return ap.parse_args(argv)


def lr_multiplier(step, total, warmup):
    if step < warmup:
        return step / max(1, warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, total - warmup))))


def token_ce_sum(hidden, targets, lm_head, chunk=128):
    keep = targets != -100
    hidden, targets = hidden[keep], targets[keep]
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, len(targets), chunk):
        total = total + checkpoint(lambda h, y: F.cross_entropy(lm_head(h).float(), y, reduction="sum"),
                                   hidden[start:start + chunk], targets[start:start + chunk], use_reentrant=False)
    return total, len(targets)


def dataset(path, tokenizer, processor):
    config = OmegaConf.create(dict(max_length=8192, pad_mode="no_padding", truncation="error",
                                   enable_thinking_default=False, shuffle=False))
    return MultiTurnSFTDataset(path, tokenizer, config, processor)


def micro_batches(reader, data, indices, prefix_only, budget):
    lengths = {i: reader.length(data[i], prefix_only) for i in indices}
    groups, current = [], []
    for i in sorted(indices, key=lambda i: -lengths[i]):
        if current and (len(current) + 1) * lengths[current[0]] > budget:
            groups.append(current)
            current = []
        current.append(i)
    return groups + [current] if current else groups


def evaluate(reader, head, data, map_targets, budget, fill=None):
    totals = {}
    loss_sum = 0.0
    with torch.no_grad():
        for group in micro_batches(reader, data, range(len(data)), True, budget):
            _, input_ids, _ = reader.forward([data[i] for i in group], prefix_only=True, fill=fill)
            for i, s in zip(group, reader.scene_states(input_ids)):
                pred = head.read(head.belief(s), map_targets[i])
                loss_sum += float(head.loss(pred, map_targets[i]))
                for k, (a, b) in head.metrics(pred, map_targets[i]).items():
                    x = totals.setdefault(k, [0, 0]); x[0] += a; x[1] += b
    return dict(map_loss=loss_sum / len(data), **{k: a / b for k, (a, b) in totals.items()},
                **{k + "_n": b for k, (a, b) in totals.items()})


SOURCES = {"mindcube": "data/smind/maps/maps", "scannet": "data/smind/maps/maps_scannet"}


def scene_states(reader, data, budget=4096):
    out = [None] * len(data)
    with torch.no_grad():
        for group in micro_batches(reader, data, range(len(data)), True, budget):
            _, input_ids, _ = reader.forward([data[i] for i in group], prefix_only=True)
            for i, s in zip(group, reader.scene_states(input_ids)):
                out[i] = s.to("cpu", torch.bfloat16)
    return out
