import argparse
import hashlib
import json
import random
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from smind.data.geometry.maps import as_parquet, gray_copy
from smind.data.geometry.scannet_map import (
    InvalidPose,
    build_map,
    gravity_frame,
    token_points,
)
from smind.model.readout import TOKEN_DIR

FRAMES = Path("data/raw/ScanNet/frames")
FRAMES_25K = Path("data/raw/ScanNet/frames_25k/scannet_frames_25k")
MAPS = Path("data/smind/maps")
SIZE = (448, 320)
TRAIN_VIEWS = [1, 2, 3, 4, 5, 6, 8, 10, 12]
DEV_VIEWS = [2, 3, 4, 6, 8, 10, 12, 16, 24]
TRAIN_PER_SCAN = 7
MIN_SHARE, MAX_SHARE = 0.15, 0.85
MAX_VIEWS, MAX_CELLS = 24, (SIZE[0] // 32) * (SIZE[1] // 32)


def dev_rooms():
    rooms = set()
    for f in ("spar_dev", "spatialladder_dev"):
        for line in open(f"data/processed/{f}.jsonl"):
            rooms.add(json.loads(line)["scene_id"].split("_")[0])
    return rooms


class Frame:

    def __init__(self, scene, fid, color, depth, pose, kd, kc, wh):
        self.scene, self.fid, self.color, self.depth_path, self.kd, self.kc, self.wh = scene, fid, color, depth, kd, kc, wh
        self.pose = np.loadtxt(pose)
        if not np.isfinite(self.pose).all():
            raise InvalidPose(f"{scene} {fid}")
        self._depth = None

    @property
    def depth(self):
        if self._depth is None:
            self._depth = np.asarray(Image.open(self.depth_path), dtype=np.float64) / 1000.0
        return self._depth

    def points(self, step):
        d = self.depth
        v, u = np.mgrid[0:d.shape[0]:step, 0:d.shape[1]:step]
        z = d[::step, ::step]
        ok = z > 0.1
        cam = np.stack([(u[ok] - self.kd[0, 2]) * z[ok] / self.kd[0, 0], (v[ok] - self.kd[1, 2]) * z[ok] / self.kd[1, 1],
                        z[ok]], -1)
        uc = (self.kc[0, 0] * cam[:, 0] / cam[:, 2] + self.kc[0, 2]) / self.wh[0]
        vc = (self.kc[1, 1] * cam[:, 1] / cam[:, 2] + self.kc[1, 2]) / self.wh[1]
        inside = (uc >= 0) & (uc < 1) & (vc >= 0) & (vc < 1)
        world = cam[inside] @ self.pose[:3, :3].T + self.pose[:3, 3]
        return world, np.stack([uc[inside], vc[inside]], -1)

    def share_in(self, other, world):
        p = (np.linalg.inv(other.pose) @ np.c_[world, np.ones(len(world))].T).T[:, :3]
        z = p[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = other.kd[0, 0] * p[:, 0] / z + other.kd[0, 2]
            v = other.kd[1, 1] * p[:, 1] / z + other.kd[1, 2]
        d = other.depth
        ok = (z > 0.1) & (u >= 0) & (u < d.shape[1]) & (v >= 0) & (v < d.shape[0])
        if not ok.any():
            return 0.0
        dz = d[v[ok].astype(int), u[ok].astype(int)]
        return float(((dz > 0) & (np.abs(dz - z[ok]) < 0.1 * z[ok])).sum() / len(world))


def s1_frames(scene):
    f = FRAMES / scene
    if not (f / "color").is_dir():
        return {}
    kd = np.loadtxt(f / "intrinsic" / "intrinsic_depth.txt")[:3, :3]
    kc = np.loadtxt(f / "intrinsic" / "intrinsic_color.txt")[:3, :3]
    meta = json.load(open(f / "intrinsic" / "meta.json"))
    out = {}
    for c in sorted((f / "color").iterdir()):
        fid = int(c.stem)
        try:
            out[fid] = Frame(scene, fid, c, f / "depth" / f"{fid}.png", f / "pose" / f"{fid}.txt", kd, kc,
                             (meta["color_width"], meta["color_height"]))
        except InvalidPose:
            pass
    return out


def s2_frames(scene):
    f = FRAMES_25K / scene
    if not (f / "color").is_dir():
        return {}
    kd = np.loadtxt(f / "intrinsics_depth.txt")[:3, :3]
    kc = np.loadtxt(f / "intrinsics_color.txt")[:3, :3]
    out = {}
    for c in sorted((f / "color").iterdir()):
        fid = int(c.stem)
        wh = Image.open(c).size
        try:
            out[fid] = Frame(scene, fid, c, f / "depth" / f"{c.stem}.png", f / "pose" / f"{c.stem}.txt", kd,
                             colour_intrinsics(kc, wh), wh)
        except InvalidPose:
            pass
    return out


def colour_intrinsics(kc, wh):
    sx, sy = wh[0] / (2 * kc[0, 2]), wh[1] / (2 * kc[1, 2])
    if abs(sx - 1) < 0.1 and abs(sy - 1) < 0.1:
        return kc
    k = kc.copy()
    k[0] *= wh[0] / 1296.0
    k[1] *= wh[1] / 968.0
    if abs(k[0, 2] / wh[0] - 0.5) > 0.1 or abs(k[1, 2] / wh[1] - 0.5) > 0.1:
        raise ValueError(f"colour intrinsics {kc[:2, 2]} do not fit a {wh} image")
    return k


def stored_image(frame, out):
    p = out / "images" / frame.scene / f"{frame.fid}.jpg"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.open(frame.color).convert("RGB").resize(SIZE, Image.BICUBIC).save(p, quality=95)
    return str(p.resolve())


def draw_sets(frames, counts, rng, per_count):
    fids = sorted(frames)
    pts = {f: frames[f].points(8)[0] for f in fids}
    pts = {f: p[rng.sample(range(len(p)), min(len(p), 600))] if len(p) else p for f, p in pts.items()}
    share = {}

    def s(a, b):
        if (a, b) not in share:
            share[a, b] = frames[a].share_in(frames[b], pts[a]) if len(pts[a]) else 0.0
        return share[a, b]

    out, seen = [], set()
    for n in counts:
        if n > len(fids):
            continue
        made = 0
        for _ in range(8 * per_count):
            if made == per_count:
                break
            chosen = [rng.choice(fids)]
            pool = [f for f in fids if f != chosen[0]]
            rng.shuffle(pool)
            while len(chosen) < n and pool:
                for f in pool:
                    links = [max(s(f, c), s(c, f)) for c in chosen]
                    if max(links) >= MIN_SHARE and max(links) <= MAX_SHARE:
                        chosen.append(f)
                        pool.remove(f)
                        break
                else:
                    break
            key = tuple(sorted(chosen))
            if len(chosen) == n and key not in seen:
                seen.add(key)
                out.append(tuple(chosen))
                made += 1
    return out


def scan_sets(job):
    scene, split, seed, out_dir = job
    frames = s1_frames(scene)
    for fid, fr in s2_frames(scene).items():
        frames.setdefault(fid, fr)
    if not frames:
        return []
    rng = random.Random(f"{seed}|{scene}")
    counts = sorted(rng.sample(TRAIN_VIEWS, TRAIN_PER_SCAN)) if split == "train" else DEV_VIEWS
    per = 1
    rows = []
    for fids in draw_sets(frames, counts, rng, per):
        fs = [frames[f] for f in fids]
        c, r, f = gravity_frame(fs[0].pose)
        grids, views = [], []
        for fr in fs:
            world, uv = fr.points(2)
            p = world - c
            views.append(token_points(np.stack([p @ r, p @ f, p[:, 2]], -1), uv, SIZE[1] // 32, SIZE[0] // 32))
            grids.append((SIZE[1] // 32, SIZE[0] // 32))
        images = [stored_image(fr, Path(out_dir)) for fr in fs]
        set_id = "x" + hashlib.sha1(f"regroup|{scene}|{fids}".encode()).hexdigest()[:15]
        m = dict(set_id=set_id, question_id=set_id, source="scannet_regroup", task="scannet_regroup", images=images,
                 n_views=len(fs), metric=True, scene=scene, frames=list(fids),
                 **build_map([], 0.0, [(fr.pose, None, None) for fr in fs]))
        rows.append((m, grids, views))
    return rows


def write_points(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = np.zeros((len(rows), MAX_VIEWS, 2), np.int16)
    points = np.full((len(rows), MAX_VIEWS, MAX_CELLS, 3), np.nan, np.float16)
    for i, (_, grids, views) in enumerate(rows):
        for k, (g, pts) in enumerate(zip(grids, views)):
            grid[i, k], points[i, k, :len(pts)] = g, pts
    np.savez_compressed(path, set_ids=np.array([m["set_id"] for m, _, _ in rows]), grid=grid, points=points)


def write_maps(rows, name, out):
    with open(out / f"{name}.jsonl", "w") as h:
        for m in rows:
            h.write(json.dumps(m) + "\n")
    as_parquet(rows, out / f"{name}.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="a new folder; an existing one is an error")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--scans", type=int, help="only the first scans of each split (quick checks)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    rooms = dev_rooms()
    scans = {p.name for p in FRAMES.iterdir() if p.is_dir()}
    if not FRAMES_25K.is_dir():
        raise SystemExit(f"{FRAMES_25K} is missing: unzip scannet_frames_25k first")
    scans |= {p.name for p in FRAMES_25K.iterdir() if p.is_dir()}
    split = {s: "dev" if s.split("_")[0] in rooms else "train" for s in sorted(scans)}
    jobs = [(s, sp, args.seed, str(out)) for s, sp in split.items()]
    if args.scans:
        jobs = [j for sp in ("train", "dev") for j in [j for j in jobs if j[1] == sp][:args.scans]]
    with Pool(48) as pool:
        done = pool.map(scan_sets, jobs, chunksize=1)
    new = {"train": [], "dev": []}
    for (_, sp, *_), rows in zip(jobs, done):
        new[sp] += rows
    for sp in new:
        new[sp].sort(key=lambda x: x[0]["set_id"])
    for f in MAPS.iterdir():
        if f.is_file() and (f.name.startswith("maps_dev") or f.name.startswith("maps_train")
                            or f.name.startswith("maps_scannet_dev")):
            shutil.copy(f, out / f.name)
    old = [json.loads(line) for line in open(MAPS / "maps_scannet_train.jsonl")]
    write_maps(old + [m for m, _, _ in new["train"]], "maps_scannet_train", out)
    dev = [m for m, _, _ in new["dev"]]
    write_maps(dev, "maps_scannetx_dev", out)
    write_maps([dict(m, images=[gray_copy(p, out) for p in m["images"]]) for m in dev], "maps_scannetx_dev_gray", out)
    write_points(new["train"], TOKEN_DIR / "points_scannetx_train.npz")
    write_points(new["dev"], TOKEN_DIR / "points_scannetx_dev.npz")
    for sp, rows in new.items():
        hist = {}
        for m, _, _ in rows:
            hist[m["n_views"]] = hist.get(m["n_views"], 0) + 1
        print(sp, len(rows), "sets from", len({m["scene"] for m, _, _ in rows}), "scans; views:", dict(sorted(hist.items())))
    print("training ScanNet sets:", len(old) + len(new["train"]))


if __name__ == "__main__":
    main()
