import glob
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
PROC = ROOT / "data/processed"
LETTERS = "ABCDEFGH"


def save(img_bytes, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    ext = (Image.open(io.BytesIO(img_bytes)).format or "PNG").lower().replace("jpeg", "jpg")
    p = folder / f"{name}.{ext}"
    p.write_bytes(img_bytes)
    return str(p)


def write(rows, name):
    with open(PROC / f"bench_{name}.jsonl", "w") as h:
        for r in rows:
            if r["answer"] not in [o[0] for o in r["options"]]:
                raise ValueError(f"{r['question_id']}: answer {r['answer']} not among {r['options']}")
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows), "questions")


def blink():
    rows = []
    for task in ["Multi-view_Reasoning", "Relative_Depth", "Spatial_Relation"]:
        df = pd.read_parquet(glob.glob(str(ROOT / f"data/raw/BLINK/{task}/val-*.parquet"))[0])
        for r in df.itertuples():
            stem = r.prompt.split("\nSelect from the following choices")[0].strip()
            choices = list(r.choices)
            imgs = [save(getattr(r, f"image_{k}")["bytes"], PROC / "images_blink", f"{r.idx}_{k}")
                    for k in range(1, 5) if getattr(r, f"image_{k}") is not None]
            rows.append(dict(source="blink", task=task, question_id=f"blink_{r.idx}", images=imgs, question=stem,
                             options=[f"{LETTERS[i]}. {c}" for i, c in enumerate(choices)], answer=r.answer.strip("()"),
                             split="val"))
    write(rows, "blink_spatial")


if __name__ == "__main__":
    blink()
