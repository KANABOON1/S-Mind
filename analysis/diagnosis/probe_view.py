import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor

from smind.model.checkpoint import SpatialCheckpoint
from smind.model.reader import SceneReader
from smind.model.readout import POINT_FAR, READOUTS, attach_token_points, load_maps
from smind.model.views import resolve, viewer_frame
from smind.train.common import dataset, micro_batches

DATA = "data/smind/sft/gpt-6-astra/perspective"
QUAD_MIN = 0.5


def collect(reader, head, layout, data, maps, specs, rows, budget):
    scene = reader.scene.cuda()
    out = []
    with torch.no_grad():
        for group in micro_batches(reader, data, rows, False, budget):
            _, input_ids, _ = reader.forward([data[i] for i in group])
            for b, i in enumerate(group):
                ids = input_ids[b, :reader.length(data[i], False)]
                pos = torch.isin(ids, scene).nonzero()[:, 0]
                starts = pos[torch.cat([torch.tensor([True], device=pos.device), pos[1:] != pos[:-1] + 1])]
                n = len(data[i]["multi_modal_inputs"]["image_grid_thw"])
                R = layout.run_length(n)
                if len(pos) != len(starts) * R or len(starts) - 1 != len(specs[i]):
                    raise ValueError(f"row {i}: {len(pos)} spatial tokens in {len(starts)} runs, {len(specs[i])} lines")
                m = maps[i]
                t = head.targets(m, "cuda")
                if "points" not in t:
                    continue
                cams = torch.stack([torch.zeros(3, device="cuda")] + [torch.cat([xy, torch.atan2(f[0], f[1])[None]])
                                                                     for xy, f in zip(t["xy"], t["facing"])])
                objs = (torch.tensor([o["xy"] for o in m["objects"]], dtype=torch.float32, device="cuda") / t["scale"]
                        if m.get("objects") else None)
                poses = resolve(specs[i], cams, objs)
                if all(p is None for p in poses):
                    continue
                grid = t["grid"]
                if len(grid) != n or len(t["seen"]) != sum(r * c for r, c in grid):
                    raise ValueError(f"row {i}: {n} images, map grid {grid}, {len(t['seen'])} map cells")
                full = torch.zeros(len(t["seen"]), 3, device="cuda")
                full[t["seen"]] = t["points"]
                lines = []
                for k, (spec, pose) in enumerate(zip(specs[i], poses)):
                    if pose is None:
                        continue
                    own = spec["image"] if spec["kind"] == "camera" else None
                    lines.append(dict(k=k, want=viewer_frame(full, pose).cpu(), own=own,
                                      kind="camera" if spec["kind"] == "camera" else "turned"))
                layers = range(len(reader.blocks))
                out.append(dict(
                    row=i, n=n, R=R, grid=grid, seen=t["seen"].cpu(), cell_view=head.cell_views(dict(t, grid=grid)).cpu(),
                    s=torch.stack([reader.outputs[j][b, pos] for j in layers]).to(torch.bfloat16).cpu(),
                    lines=lines))
            reader.outputs = {}
    return out


def view_of(r):
    return dict(n_views=r["n"], grid=r["grid"], view_index=torch.arange(1, r["n"], device="cuda"))


def states(r, arm, k):
    R = r["R"]
    s = r["s"][:, :R] if arm == "s_0" else r["s"][:, (k + 1) * R:(k + 2) * R]
    s = s.cuda().float()
    return torch.zeros_like(s) if arm == "zero" else s


def new_head(arm, layers, hidden, layer_logits, seed, layout):
    torch.manual_seed(seed)
    head = READOUTS["geometry_group"](layers, hidden, m=layout.per_image, c=layout.camera)
    del head.layer_logits
    head.register_buffer("layer_logits", layer_logits.clone())
    return head.cuda()


def predict(head, r, arm, k):
    return head.read(head.belief(states(r, arm, k)), view_of(r))["points"].float()


def fit(arm, rows, targets, layers, hidden, layer_logits, epochs, seed, layout):
    head = new_head(arm, layers, hidden, layer_logits, seed, layout)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    order, rng = list(range(len(rows))), random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), 64):
            loss = 0.0
            for i in order[start:start + 64]:
                r = rows[i]
                for line, want in zip(r["lines"], targets[i]):
                    if want is None:
                        continue
                    got = predict(head, r, arm, line["k"])[r["seen"].cuda()]
                    loss = loss + F.smooth_l1_loss(got, want.cuda()[r["seen"].cuda()], beta=0.1, reduction="none").sum(-1).mean()
            opt.zero_grad(); (loss / 64).backward(); opt.step()
    return head.eval()


def counts(got, want, seen, cell_view, line):
    got, want, cv = got[seen], want[seen], cell_view[seen]
    own = (cv == line["own"]) if line["own"] is not None else torch.zeros_like(cv, dtype=torch.bool)
    far = (got - want).norm(dim=-1) <= POINT_FAR
    def quad(p):
        return torch.round(torch.atan2(p[:, 0], p[:, 1]) / (math.pi / 2)).long() % 4
    qmask = want[:, :2].norm(dim=-1) >= QUAD_MIN
    qok = quad(got) == quad(want)
    out = {}
    for split, sel in (("all", torch.ones_like(own)), ("own_image", own), ("other_image", ~own)):
        for tag in ("all", line["kind"]):
            if sel.any():
                out[("view_far", split, tag)] = (int(far[sel].sum()), int(sel.sum()))
            q = sel & qmask
            if q.any():
                out[("quad", split, tag)] = (int(qok[q].sum()), int(q.sum()))
                for d in range(4):
                    qd = q & (quad(want) == d)
                    if qd.any():
                        out[(f"quad{d}", split, tag)] = (int(qok[qd].sum()), int(qd.sum()))
    return out


def score(per_row):
    keys = sorted({k for r in per_row for k in r if not k[0].startswith("quad") or k[0] == "quad"})
    bal = sorted({k[1:] for r in per_row for k in r if k[0] == "quad"})
    def acc(rows, key):
        a = sum(r[key][0] for r in rows if key in r)
        b = sum(r[key][1] for r in rows if key in r)
        return a / b if b else None
    def bal_acc(rows, rest):
        xs = [acc(rows, (f"quad{d}",) + rest) for d in range(4)]
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None
    rng, out = random.Random(0), {}
    samples = [[per_row[rng.randrange(len(per_row))] for _ in per_row] for _ in range(1000)]
    for name, f, items in ([("/".join(k), lambda rows, k=k: acc(rows, k), sum(r[k][1] for r in per_row if k in r)) for k in keys]
                           + [("/".join(("quad_bal",) + k), lambda rows, k=k: bal_acc(rows, k),
                               sum(r[("quad",) + k][1] for r in per_row if ("quad",) + k in r)) for k in bal]):
        value = f(per_row)
        if value is None:
            continue
        boot = sorted(x for x in (f(s) for s in samples) if x is not None)
        out[name] = dict(acc=round(value, 4), ci=[round(boot[len(boot) // 40], 4), round(boot[-1 - len(boot) // 40], 4)],
                         items=items)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="the probe tokens of geometry_probe.sh (<its OUT>/probe/ckpt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=DATA, help="perspective-stage rows with a held-out split (README, Data)")
    ap.add_argument("--train_rows", type=int, default=1500, help="train rows the readouts are fit on (random, seed 17)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--heldout_rows", type=int, help="the first held-out rows only (quick checks)")
    ap.add_argument("--budget", type=int, default=8192)
    args = ap.parse_args()
    ckpt = SpatialCheckpoint.load(args.ckpt)
    layout = ckpt.layout
    processor = AutoProcessor.from_pretrained(ckpt.path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(ckpt.path, dtype=torch.bfloat16,
                                                        attn_implementation="sdpa", local_files_only=True).cuda().eval()
    _, template = ckpt.readout()
    reader = SceneReader(model, ckpt.token_ids(processor.tokenizer), processor.tokenizer.pad_token_id)
    layers, hidden = len(reader.blocks), model.config.text_config.hidden_size
    layer_logits = template.layer_logits.detach().float()
    template = template.cuda()
    rows = {}
    for split, n in (("train", args.train_rows), ("heldout", args.heldout_rows)):
        data = dataset(f"{args.data}/{split}_z.parquet", processor.tokenizer, processor)
        maps = attach_token_points(load_maps(f"{args.data}/{split}_maps.jsonl"))
        specs = json.loads(Path(f"{args.data}/{split}_targets.json").read_text())
        keep = (sorted(random.Random(17).sample(range(len(data)), n)) if split == "train" else
                list(range(min(n or len(data), len(data)))))
        rows[split] = collect(reader, template, layout, data, maps, specs, keep, args.budget)
        print(split, len(rows[split]), "rows with a view target,", sum(len(r["lines"]) for r in rows[split]), "lines",
              flush=True)
    del model, reader
    torch.cuda.empty_cache()
    train, held = rows["train"], rows["heldout"]
    true_targets = [[l["want"] for l in r["lines"]] for r in train]
    rng, by_grid = random.Random(17), {}
    for i, r in enumerate(train):
        by_grid.setdefault(tuple(r["grid"]), []).append(i)
    shuffled = []
    for i, r in enumerate(train):
        partners = [x for x in by_grid[tuple(r["grid"])] if x != i]
        if not partners:
            shuffled.append([None] * len(r["lines"]))
            continue
        other = [l["want"] for l in train[rng.choice(partners)]["lines"]]
        shuffled.append([other[q % len(other)] for q in range(len(r["lines"]))])
    result = dict(ckpt=str(ckpt.path), data=args.data, train_rows=len(train), heldout_rows=len(held),
                  heldout_lines=sum(len(r["lines"]) for r in held), arms={})
    for arm, targets in (("s_k", true_targets), ("s_0", true_targets),
                         ("zero", true_targets), ("shuffled", shuffled)):
        fit_arm = "s_k" if arm == "shuffled" else arm
        head = fit(fit_arm, train, targets, layers, hidden, layer_logits, args.epochs, 17, layout)
        per_row, wrong = [], []
        with torch.no_grad():
            for r in held:
                c, w = {}, {}
                for line in r["lines"]:
                    got = predict(head, r, fit_arm, line["k"]).cpu()
                    for key, (a, b) in counts(got, line["want"], r["seen"], r["cell_view"], line).items():
                        c[key] = (c.get(key, (0, 0))[0] + a, c.get(key, (0, 0))[1] + b)
                    if arm == "s_k":
                        for other in r["lines"]:
                            if other["k"] != line["k"] and not torch.allclose(other["want"], line["want"]):
                                for key, (a, b) in counts(got, other["want"], r["seen"], r["cell_view"], line).items():
                                    w[key] = (w.get(key, (0, 0))[0] + a, w.get(key, (0, 0))[1] + b)
                per_row.append(c)
                wrong.append(w)
        result["arms"][arm] = score(per_row)
        if arm == "s_k":
            result["arms"]["wrong_view"] = score([w for w in wrong if w])
            result["per_row_s_k"] = [{"/".join(k): v for k, v in c.items()} for c in per_row]
        print(arm, {k: v["acc"] for k, v in result["arms"][arm].items() if k.endswith("/all")}, flush=True)
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
