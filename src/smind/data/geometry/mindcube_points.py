import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from smind.data.geometry.maps import GEOMETRY_DIR, MAPS_DIR
from smind.data.geometry.scannet_map import MAX_CELLS, MAX_VIEWS, TOKEN, token_points
from smind.model.readout import TOKEN_DIR

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "third_party/vggt"))
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images_square  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


def support_plane(p, rng, iters=300, tol=0.02):
    best, best_n = None, 0
    scale = np.median(np.linalg.norm(p, axis=1))
    for _ in range(iters):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n)
        if n[1] > 0:
            n = -n
        if -n[1] < math.cos(math.radians(50)):
            continue
        d = -n @ a
        if d <= 0:
            continue
        inliers = np.abs(p @ n + d) < tol * scale
        if inliers.sum() > best_n:
            best, best_n = (n, d), inliers.sum()
    if best is None:
        return None
    n, _ = best
    inl = np.abs(p @ best[0] + best[1]) < tol * scale
    q = p[inl]
    centre = q.mean(0)
    _, _, vt = np.linalg.svd(q - centre)
    n = vt[-1] if vt[-1][1] < 0 else -vt[-1]
    return n, -n @ centre


def frame_of(m, pred, rng):
    pts = pred["world_points"][0].float().cpu().numpy()
    conf = pred["world_points_conf"][0].float().cpu().numpy()
    extr, _ = pose_encoding_to_extri_intri(pred["pose_enc"], pred["images"].shape[-2:])
    extr = extr[0].float().cpu().numpy()
    keep = conf > np.percentile(conf, 50)
    p = pts[keep]
    if len(p) < 1000:
        return "few points"
    plane = support_plane(p[rng.choice(len(p), min(len(p), 20000), replace=False)], rng)
    if plane is None:
        return "no support plane"
    up, d = plane
    fwd = np.array([0.0, 0.0, 1.0]) - up * up[2]
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    flat = lambda v: np.array([v @ right, v @ fwd])
    ratios, agree = [], []
    for k, v in enumerate(m["views"]):
        if k == 0 or k >= len(extr):
            continue
        r, t = extr[k][:, :3], extr[k][:, 3]
        c, f = -r.T @ t, r.T @ np.array([0.0, 0.0, 1.0])
        dist = np.linalg.norm(flat(c))
        target = np.hypot(*v["xy"])
        if dist > 1e-6 and target > 1e-6:
            ratios.append(target / dist)
        fv = flat(f)
        agree.append(np.dot(fv / max(np.linalg.norm(fv), 1e-9), np.array(v["facing"], float)) > math.cos(math.pi / 4))
    if not all(agree):
        return "VGGT cameras disagree with the template"
    return right, fwd, up, d, float(np.median(ratios)) if ratios else None, p


def write_token_points(model):
    rng = np.random.default_rng(17)
    for split in ("train", "dev"):
        maps = {m["set_id"]: m for m in (json.loads(l) for l in open(MAPS_DIR / f"maps_{split}.jsonl"))}
        ids, grids, points, why = [], [], [], {}
        for i, (sid, m) in enumerate(sorted(maps.items())):
            images, boxes = load_and_preprocess_images_square(m["images"], target_size=518)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(images.cuda())
            frame = frame_of(m, pred, rng)
            key = frame if isinstance(frame, str) else "ok" if frame[4] is not None else "ok, VGGT unit"
            why[key] = why.get(key, 0) + 1
            if not isinstance(frame, str):
                right, fwd, up, _, s, _ = frame
                s = 1.0 if s is None else s
                pts = pred["world_points"][0].float().cpu().numpy()
                conf = pred["world_points_conf"][0].float().cpu().numpy()
                grid = np.zeros((MAX_VIEWS, 2), np.int16)
                out = np.full((MAX_VIEWS, MAX_CELLS, 3), np.nan, np.float16)
                for k, ((x1, y1, x2, y2, _, _), path) in enumerate(zip(boxes.tolist(), m["images"])):
                    w, h = Image.open(path).size
                    rows, cols = h // TOKEN, w // TOKEN
                    box = (slice(round(y1), round(y2)), slice(round(x1), round(x2)))
                    q, c = pts[k][box], conf[k][box]
                    keep = c > np.percentile(c, 30)
                    vv, uu = np.nonzero(keep)
                    frac = np.stack([(uu + 0.5) / q.shape[1], (vv + 0.5) / q.shape[0]], -1)
                    floor = np.stack([q[keep] @ right, q[keep] @ fwd, q[keep] @ up], -1) * s
                    grid[k] = rows, cols
                    out[k, :rows * cols] = token_points(floor, frac, rows, cols)
                ids.append(sid); grids.append(grid); points.append(out)
            if i % 200 == 0:
                print(split, i, len(maps), why, flush=True)
        TOKEN_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(TOKEN_DIR / f"points_mindcube_{split}.npz", set_ids=np.array(ids),
                            grid=np.stack(grids), points=np.stack(points))
        print(f"{split}: points for {len(ids)}/{len(maps)} sets {why}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vggt", default=str(ROOT / "models/VGGT-1B/model.pt"), help="VGGT-1B weights (model.pt)")
    args = ap.parse_args()
    GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
    model = VGGT()
    model.load_state_dict(torch.load(args.vggt, map_location="cpu"))
    write_token_points(model.cuda().eval())


if __name__ == "__main__":
    main()
