import argparse
import json
import os
import struct
import subprocess
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/raw/ScanNet"
BASE = None
GT_TYPES = ["_vh_clean.ply", "_vh_clean.segs.json", "_vh_clean.aggregation.json", ".txt"]


def fetch(url, dst, tries=5):
    if dst.exists():
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    for k in range(tries):
        r = subprocess.run(["curl", "-sSL", "--fail", "-C", "-", "-o", str(tmp), url], capture_output=True)
        if r.returncode == 0:
            tmp.rename(dst)
            return
        time.sleep(10 * (k + 1))
    raise RuntimeError(f"download failed: {url}: {r.stderr.decode()[-300:]}")


def extract(sens, out, frames):
    want = set(frames)
    with open(sens, "rb") as f:
        version, = struct.unpack("I", f.read(4))
        assert version == 4, version
        strlen, = struct.unpack("Q", f.read(8))
        f.read(strlen)
        mats = [np.frombuffer(f.read(64), dtype=np.float32).reshape(4, 4) for _ in range(4)]
        color_comp, depth_comp = struct.unpack("ii", f.read(8))
        cw, ch, dw, dh = struct.unpack("IIII", f.read(16))
        depth_shift, = struct.unpack("f", f.read(4))
        n, = struct.unpack("Q", f.read(8))
        assert color_comp == 2 and depth_comp == 1, (color_comp, depth_comp)
        assert max(want) < n, (max(want), n)
        for d in ("color", "depth", "pose", "intrinsic"):
            (out / d).mkdir(parents=True, exist_ok=True)
        names = ["intrinsic_color", "extrinsic_color", "intrinsic_depth", "extrinsic_depth"]
        for name, m in zip(names, mats):
            np.savetxt(out / "intrinsic" / f"{name}.txt", m)
        (out / "intrinsic" / "meta.json").write_text(json.dumps(dict(
            color_width=cw, color_height=ch, depth_width=dw, depth_height=dh, depth_shift=depth_shift, num_frames=n)))
        for i in range(n):
            pose = np.frombuffer(f.read(64), dtype=np.float32).reshape(4, 4)
            f.read(16)
            cs, ds = struct.unpack("QQ", f.read(16))
            if i not in want:
                f.seek(cs + ds, os.SEEK_CUR)
                continue
            (out / "color" / f"{i}.jpg").write_bytes(f.read(cs))
            depth = np.frombuffer(zlib.decompress(f.read(ds)), dtype=np.uint16).reshape(dh, dw)
            Image.fromarray(depth).save(out / "depth" / f"{i}.png")
            np.savetxt(out / "pose" / f"{i}.txt", pose)
            want.discard(i)
            if not want:
                break
    assert not want, sorted(want)[:5]


def scene(name, frames, keep_sens=False):
    out = OUT / "frames" / name
    done = out / ".done"
    frames = [i for i in frames if not (out / "color" / f"{i}.jpg").exists()]
    if done.exists() and not frames:
        return name, "cached"
    gt = OUT / "scans" / name
    gt.mkdir(parents=True, exist_ok=True)
    for t in GT_TYPES:
        fetch(f"{BASE}v2/scans/{name}/{name}{t}", gt / f"{name}{t}")
    sens = OUT / "tmp" / f"{name}.sens"
    sens.parent.mkdir(parents=True, exist_ok=True)
    fetch(f"{BASE}v1/scans/{name}/{name}.sens", sens)
    extract(sens, out, frames)
    if not keep_sens:
        sens.unlink()
    done.write_text(str(len(frames)))
    return name, len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="JSON file {scene: [frame indices]}: spar_frames.json (smind.data.build_spar select) or sl_frames.json (smind.data.build)")
    ap.add_argument("--base_url", required=True, help="the ScanNet download URL you received with access approval")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    global BASE
    BASE = args.base_url.rstrip("/") + "/"
    todo = json.loads(Path(args.frames).read_text())
    items = list(todo.items())[: args.limit]
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(scene, s, fr) for s, fr in items]
        for k, fu in enumerate(futs):
            try:
                s, n = fu.result()
                print(f"[{k + 1}/{len(items)}] {s}: {n} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            except Exception as e:
                print(f"[{k + 1}/{len(items)}] FAILED {items[k][0]}: {e}", flush=True)


if __name__ == "__main__":
    main()
