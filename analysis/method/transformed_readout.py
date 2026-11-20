import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from smind.model.checkpoint import SpatialCheckpoint
from smind.model.reader import SceneReader
from smind.model.readout import attach_token_points, load_maps
from smind.model.views import camera_poses, viewer_frame
from smind.train.common import dataset, evaluate, micro_batches
from smind.train.perspective import DATA, DEV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="an S-Mind checkpoint after the perspective stage or RL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--micro_tokens", type=int, default=8192)
    ap.add_argument("--data", default=DATA, help="perspective-stage rows with a held-out split (README, Data)")
    ap.add_argument("--dev", nargs="+", default=DEV, help="the fusion stage's dev image sets, read off z0")
    args = ap.parse_args()
    ckpt = SpatialCheckpoint.load(args.ckpt)
    if ckpt.insertion is None:
        raise ValueError(f"{ckpt.path} is a fusion-stage checkpoint")
    processor = AutoProcessor.from_pretrained(ckpt.path, local_files_only=True)
    tok = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(ckpt.path, dtype=torch.bfloat16,
                                                        attn_implementation="sdpa", local_files_only=True).cuda().eval()
    _, head = ckpt.readout()
    head = head.cuda().eval()
    layout, scene_ids = ckpt.layout, ckpt.token_ids(tok)
    token_ids = torch.tensor(scene_ids, device="cuda")
    data = dataset(f"{args.data}/heldout_z.parquet", tok, processor)
    maps = attach_token_points(load_maps(f"{args.data}/heldout_maps.jsonl"))
    specs = json.loads(Path(f"{args.data}/heldout_targets.json").read_text())
    reader = SceneReader(model, scene_ids, tok.pad_token_id)

    z0 = {}
    for path in args.dev:
        dev = dataset(path + ".parquet", tok, processor)
        targets = [head.targets(m, "cuda") for m in attach_token_points(load_maps(path + ".jsonl"))]
        z0[Path(path).name] = evaluate(reader, head, dev, targets, args.micro_tokens)["point_far"]
        reader.outputs = {}

    n = dict(lines=0, hit_own=0, true_lines=0, hit_true=0, rows=0)
    with torch.no_grad():
        for group in micro_batches(reader, data, range(len(data)), False, args.micro_tokens):
            items = [data[i] for i in group]
            _, input_ids, _ = reader.forward(items)
            for b, i in enumerate(group):
                t = head.targets(maps[i], "cuda")
                pos = torch.isin(input_ids[b, :len(items[b]["input_ids"])], token_ids).nonzero()[:, 0]
                states = torch.stack([reader.outputs[j][b, pos] for j in range(len(reader.blocks))])
                grid = [(int(h) // 2, int(w) // 2) for _, h, w in items[b]["multi_modal_inputs"]["image_grid_thw"].tolist()]
                view = dict(n_views=len(grid), grid=grid, view_index=torch.arange(1, len(grid), device="cuda"))
                R = layout.run_length(view["n_views"])
                read0 = head.read(head.belief(states[:, :R]).float(), view)
                p0, cams0 = read0["points"].float(), camera_poses(read0["camera"].float())
                own = torch.stack([viewer_frame(p0, pose) for pose in cams0])
                mask = t["seen"] if "seen" in t else torch.ones(len(p0), dtype=torch.bool, device="cuda")
                truth = "points" in t and len(t["points"]) and "xy" in t
                n["rows"] += 1
                if truth:
                    pts = t["points"].float()
                    cams_t = camera_poses(torch.cat([t["xy"], t["facing"]], -1).float())
                    seen_true = torch.stack([viewer_frame(pts, pose) for pose in cams_t])
                if states.shape[1] // R - 1 != len(specs[i]):
                    raise ValueError(f"row {i}: {states.shape[1] // R - 1} runs, {len(specs[i])} view targets")
                for k, spec in enumerate(specs[i]):
                    if spec["kind"] != "camera":
                        continue
                    cam = spec["image"]
                    got = head.read(head.belief(states[:, (k + 1) * R:(k + 2) * R]).float(), view)["points"].float()[mask]
                    n["lines"] += 1
                    n["hit_own"] += int((got[None] - own[:, mask]).norm(dim=-1).mean(-1).argmin() == cam)
                    if truth:
                        n["true_lines"] += 1
                        n["hit_true"] += int((got[None] - seen_true).norm(dim=-1).mean(-1).argmin() == cam)
            reader.outputs = {}
    row = dict(ckpt=str(ckpt.path), **n, z0_point_far_pct={k: 100 * v for k, v in z0.items()},
               transformed_pct=100 * n["hit_own"] / max(n["lines"], 1),
               transformed_true_pct=100 * n["hit_true"] / max(n["true_lines"], 1))
    print(json.dumps(row), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(row, indent=1) + "\n")


if __name__ == "__main__":
    main()
