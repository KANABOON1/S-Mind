import argparse
import hashlib
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from smind.data.build import repo_path
from smind.data.geometry.maps import GEOMETRY_DIR, MAPS_DIR, as_parquet, gray_copy
from smind.model.readout import TOKEN_DIR

SCANNET = Path("data/raw/ScanNet")
SKIP = {"floor", "ceiling", "object"}
MIN_CELLS = 12
DEPTH_TOL = 0.10


def read_ply_xyz(path):
    with open(path, "rb") as f:
        header = []
        while header[-1:] != ["end_header"]:
            header.append(f.readline().decode().strip())
        props, in_vertex, n = [], False, 0
        for line in header:
            if line.startswith("element"):
                in_vertex = line.startswith("element vertex")
                n = int(line.split()[2]) if in_vertex else n
            elif line.startswith("property") and in_vertex:
                _, kind, name = line.split()
                props.append((name, {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4"}[kind]))
        v = np.frombuffer(f.read(n * np.dtype(props).itemsize), dtype=props, count=n)
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)


def instances(scene):
    d = SCANNET / "scans" / scene
    xyz = read_ply_xyz(d / f"{scene}_vh_clean.ply")
    seg = np.array(json.load(open(d / f"{scene}_vh_clean.segs.json"))["segIndices"])
    order = np.argsort(seg, kind="stable")
    sorted_seg = seg[order]
    out = []
    for g in json.load(open(d / f"{scene}_vh_clean.aggregation.json"))["segGroups"]:
        idx = np.concatenate([order[np.searchsorted(sorted_seg, s):np.searchsorted(sorted_seg, s, "right")]
                              for s in g["segments"]])
        out.append((g["label"].lower(), xyz[idx]))
    floor = [p[:, 2] for name, p in out if name == "floor"]
    floor_z = np.median(np.concatenate(floor)) if floor else np.percentile(xyz[:, 2], 1)
    return out, float(floor_z)


class InvalidPose(ValueError):
    pass


def camera(scene, frame):
    f = SCANNET / "frames" / scene
    pose = np.loadtxt(f / "pose" / f"{frame}.txt")
    if not np.isfinite(pose).all():
        raise InvalidPose(f"{scene} frame {frame}")
    k = np.loadtxt(f / "intrinsic" / "intrinsic_depth.txt")[:3, :3]
    depth = np.asarray(Image.open(f / "depth" / f"{frame}.png"), dtype=np.float64) / 1000.0
    return pose, k, depth


def visible_cells(points, cam):
    pose, k, depth = cam
    p = (np.linalg.inv(pose) @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    z = p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = k[0, 0] * p[:, 0] / z + k[0, 2], k[1, 1] * p[:, 1] / z + k[1, 2]
    h, w = depth.shape
    ok = (z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[ok].astype(int), v[ok].astype(int), z[ok]
    d = depth[v, u]
    hit = (d > 0) & (np.abs(d - z) < DEPTH_TOL)
    return len(set(zip((u[hit] // 8).tolist(), (v[hit] // 8).tolist())))


def gravity_frame(pose):
    f = pose[:3, 2].copy()
    f[2] = 0
    f /= np.linalg.norm(f)
    return pose[:3, 3], np.array([f[1], -f[0], 0.0]), f


def footprint(xy):
    lo, hi = np.percentile(xy, 2, 0), np.percentile(xy, 98, 0)
    xy = xy[((xy >= lo) & (xy <= hi)).all(1)]
    mean = xy.mean(0)
    _, vecs = np.linalg.eigh(np.cov((xy - mean).T))
    axes = vecs[:, ::-1]
    proj = (xy - mean) @ axes
    a, b = proj.min(0), proj.max(0)
    return mean + axes @ ((a + b) / 2), b - a, math.atan2(axes[1, 0], axes[0, 0]) % math.pi


def build_map(objs, floor_z, cams):
    c, r, f = gravity_frame(cams[0][0])
    ego = lambda p: np.stack([(p[..., :2] - c[:2]) @ r[:2], (p[..., :2] - c[:2]) @ f[:2]], -1)
    objects = []
    for name, points in objs:
        if name in SKIP or len(points) < 20:
            continue
        if max(visible_cells(points, cam) for cam in cams) < MIN_CELLS:
            continue
        center, size, yaw = footprint(ego(points))
        z = np.percentile(points[:, 2], [2, 98]) - floor_z
        objects.append(dict(name=name, xy=np.round(center, 3).tolist(), size=np.round(size, 3).tolist(),
                            yaw=round(yaw, 3), z=[round(max(float(z[0]), 0.0), 3), round(float(z[1]), 3)]))
    views = []
    for pose, _, _ in cams:
        cv, _, fv = gravity_frame(pose)
        views.append(dict(xy=np.round(ego(cv), 3).tolist(),
                          facing=[round(float(fv[:2] @ r[:2]), 4), round(float(fv[:2] @ f[:2]), 4)]))
    return dict(objects=objects, views=views)


def image_sets(split):
    sets = {}
    for source in ("spar", "spatialladder"):
        for line in open(f"data/processed/{source}_{split}.jsonl"):
            rec = json.loads(line)
            images = rec["images"]
            raw = rec["spar_image"] if source == "spar" else rec["raw_images"]
            key = tuple(images)
            if key not in sets:
                sets[key] = dict(set_id=hashlib.sha1("|".join(map(repo_path, key)).encode()).hexdigest()[:16],
                                 question_id=rec["question_id"], source=source, task=source, scene=rec["scene_id"],
                                 frames=[int(Path(p).stem) for p in raw], images=images, n_views=len(images))
    return list(sets.values())


MAX_VIEWS = 8
TOKEN = 32
MAX_CELLS = 300


def token_points(points, valid, rows, cols, min_share=0.25):
    out = np.full((rows * cols, 3), np.nan, np.float32)
    if not len(points):
        return out
    cell = (np.floor(valid[:, 1] * rows).astype(int) * cols + np.floor(valid[:, 0] * cols).astype(int))
    order = np.argsort(cell, kind="stable")
    cell, points = cell[order], points[order]
    starts = np.r_[0, np.nonzero(np.diff(cell))[0] + 1]
    need = min_share * len(order) / (rows * cols)
    for a, b in zip(starts, np.r_[starts[1:], len(cell)]):
        if b - a >= need:
            out[cell[a]] = np.median(points[a:b], axis=0)
    return out


def scene_token_points(group):
    scene, rows_ = group
    meta = json.load(open(SCANNET / "frames" / scene / "intrinsic" / "meta.json"))
    kc = np.loadtxt(SCANNET / "frames" / scene / "intrinsic" / "intrinsic_color.txt")[:3, :3]
    out = []
    for row in rows_:
        try:
            cams = [camera(scene, fr) for fr in row["frames"]]
        except InvalidPose:
            continue
        c, r, f = gravity_frame(cams[0][0])
        grids, views = [], []
        for (pose, k, depth), image in zip(cams, row["images"]):
            w, h = Image.open(image).size
            rows, cols = h // TOKEN, w // TOKEN
            v, u = np.mgrid[0:depth.shape[0]:2, 0:depth.shape[1]:2]
            z = depth[::2, ::2]
            ok = z > 0.1
            cam = np.stack([(u[ok] - k[0, 2]) * z[ok] / k[0, 0], (v[ok] - k[1, 2]) * z[ok] / k[1, 1], z[ok]], -1)
            uc = (kc[0, 0] * cam[:, 0] / cam[:, 2] + kc[0, 2]) / meta["color_width"]
            vc = (kc[1, 1] * cam[:, 1] / cam[:, 2] + kc[1, 2]) / meta["color_height"]
            inside = (uc >= 0) & (uc < 1) & (vc >= 0) & (vc < 1)
            p = cam[inside] @ pose[:3, :3].T + pose[:3, 3] - c
            floor = np.stack([p @ r, p @ f, p[:, 2]], -1)
            views.append(token_points(floor, np.stack([uc[inside], vc[inside]], -1), rows, cols))
            grids.append((rows, cols))
        out.append((row["set_id"], grids, views))
    return out


def write_token_points():
    for split in ("train", "dev"):
        maps = {json.loads(l)["set_id"] for l in open(MAPS_DIR / f"maps_scannet_{split}.jsonl")}
        by_scene = {}
        for row in image_sets(split):
            if row["set_id"] in maps:
                by_scene.setdefault(row["scene"], []).append(row)
        with Pool(32) as pool:
            rows = sorted((x for xs in pool.imap_unordered(scene_token_points, sorted(by_scene.items())) for x in xs),
                          key=lambda x: x[0])
        grid = np.zeros((len(rows), MAX_VIEWS, 2), np.int16)
        points = np.full((len(rows), MAX_VIEWS, MAX_CELLS, 3), np.nan, np.float16)
        for i, (_, grids, views) in enumerate(rows):
            for k, (g, pts) in enumerate(zip(grids, views)):
                grid[i, k], points[i, k, :len(pts)] = g, pts
        TOKEN_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(TOKEN_DIR / f"points_scannet_{split}.npz",
                            set_ids=np.array([x[0] for x in rows]), grid=grid, points=points)
        cells = grid[..., 0].astype(int) * grid[..., 1]
        seen = np.isfinite(points[..., 0]).sum(-1)
        print(f"{split}: {len(rows)} sets; seen tokens per image {seen[cells > 0].mean():.0f} of {cells[cells > 0].mean():.0f}",
              flush=True)


def scene_maps(group):
    scene, rows = group
    objs, floor_z = instances(scene)
    out = []
    for row in rows:
        keep = {k: v for k, v in row.items() if k not in ("scene", "frames")}
        try:
            cams = [camera(scene, fr) for fr in row["frames"]]
        except InvalidPose:
            out.append(dict(keep, invalid_pose=True))
            continue
        out.append(dict(keep, metric=True, **build_map(objs, floor_z, cams)))
    return out


def main():
    for split in ("train", "dev"):
        by_scene = {}
        for row in image_sets(split):
            by_scene.setdefault(row["scene"], []).append(row)
        with Pool(48) as pool:
            rows = [m for ms in pool.imap_unordered(scene_maps, sorted(by_scene.items())) for m in ms]
        rows.sort(key=lambda m: m["set_id"])
        invalid = [m for m in rows if m.get("invalid_pose")]
        empty = [m for m in rows if not m.get("invalid_pose") and not m["objects"]]
        rows = [m for m in rows if not m.get("invalid_pose") and m["objects"]]
        variants = [("", rows)] + ([("_gray", [dict(m, images=[gray_copy(p) for p in m["images"]]) for m in rows])]
                                   if split == "dev" else [])
        for suffix, rs in variants:
            name = f"maps_scannet_{split}{suffix}"
            with open(MAPS_DIR / f"{name}.jsonl", "w") as f:
                for m in rs:
                    f.write(json.dumps(m) + "\n")
            as_parquet(rs, MAPS_DIR / f"{name}.parquet")
        counts = [len(m["objects"]) for m in rows]
        print(f"{split}: {len(rows)} sets (dropped: {len(invalid)} with an invalid pose, {len(empty)} without a "
              f"visible object); objects per map "
              f"median {int(np.median(counts))}, 95th {int(np.percentile(counts, 95))}, max {max(counts)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--token_points", action="store_true",
                    help="only write data/smind/geometry/token/points_scannet_<split>.npz")
    args = ap.parse_args()
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
    if args.token_points:
        write_token_points()
    else:
        main()
