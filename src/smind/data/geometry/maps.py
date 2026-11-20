import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from smind.data.build import repo_path

RAW = "third_party/MindCube/data/scaffold/all/MindCube_train.jsonl"
MAPS_DIR = Path("data/smind/maps")
GEOMETRY_DIR = Path("data/smind/geometry")
K = 16
FLAT_TOKENS = [f"<|scene_{i}|>" for i in range(K)]
FACING = {"up": (0, -1), "right": (1, 0), "down": (0, 1), "left": (-1, 0)}


def insert_scene(user_text):
    cut = user_text.rfind("<image>") + len("<image>")
    if cut < len("<image>"):
        raise ValueError("no image placeholder")
    return user_text[:cut] + "".join(FLAT_TOKENS) + user_text[cut:]


def canonical_map(cogmap):
    views = cogmap["views"]
    fx, fy = FACING[views[0]["facing"]]
    rx, ry = -fy, fx
    x0, y0 = views[0]["position"]

    def ego(p):
        dx, dy = p[0] - x0, p[1] - y0
        return [dx * rx + dy * ry, dx * fx + dy * fy]

    def ego_dir(name):
        vx, vy = FACING[name]
        return [vx * rx + vy * ry, vx * fx + vy * fy]

    return dict(objects=[dict(name=o["name"].lower(), xy=ego(o["position"])) for o in cogmap["objects"]],
                views=[dict(xy=ego(v["position"]), facing=ego_dir(v["facing"])) for v in views])


def map_rows(split, raw, gray=False):
    rows, seen = [], set()
    for line in open(f"data/processed/mindcube_{split}.jsonl"):
        rec = json.loads(line)
        key = tuple(rec["images"])
        if key in seen:
            continue
        seen.add(key)
        m = canonical_map(json.loads(raw[rec["question_id"]]["grounded_cogmap"]))
        images = [gray_copy(p) for p in rec["images"]] if gray else rec["images"]
        rows.append(dict(set_id=hashlib.sha1("|".join(map(repo_path, key)).encode()).hexdigest()[:16],
                         question_id=rec["question_id"],
                         source="mindcube", task=rec["task"], images=images, n_views=len(images), metric=False, **m))
    return rows


def gray_copy(path, root=MAPS_DIR):
    out = Path(root) / "gray" / (hashlib.sha1(repo_path(path).encode()).hexdigest()[:20] + ".png")
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        size = Image.open(path).size
        Image.new("RGB", size, (128, 128, 128)).save(out)
    return str(out.resolve())


def as_parquet(rows, path):
    pd.DataFrame([dict(messages=[dict(role="user", content="<image>" * r["n_views"] + "".join(FLAT_TOKENS)),
                                 dict(role="assistant", content="OK")],
                       images=[dict(image=p) for p in r["images"]], enable_thinking=False, question_id=r["set_id"],
                       source=r["source"] + "_map", task=r["task"]) for r in rows]).to_parquet(path)


def maps():
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    raw = {}
    for line in open(RAW):
        r = json.loads(line)
        raw[r["id"]] = r
    for split, gray in [("train", False), ("dev", False), ("dev", True)]:
        rows = map_rows(split, raw, gray)
        name = f"maps_{split}" + ("_gray" if gray else "")
        with open(MAPS_DIR / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        as_parquet(rows, MAPS_DIR / f"{name}.parquet")
        print(name, len(rows))


def main():
    argparse.ArgumentParser().parse_args()
    maps()


if __name__ == "__main__":
    main()
