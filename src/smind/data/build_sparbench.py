import glob
import io
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
MCA = ["obj_spatial_relation_oo", "obj_spatial_relation_oc_mv", "obj_spatial_relation_oo_mv", "spatial_imagination_oc",
       "spatial_imagination_oo", "spatial_imagination_oc_mv", "spatial_imagination_oo_mv", "position_matching",
       "camera_motion_infer", "distance_infer_center_oo", "distance_infer_center_oo_mv"]
OPTION = re.compile(r"^([A-Z])\. ?(.*)$")


def split(question):
    lines = question.strip().split("\n")
    if not lines[-1].startswith("Your answer can only include"):
        raise ValueError(f"no answer instruction: {question[-120:]!r}")
    lines = lines[:-1]
    first = next(i for i, l in enumerate(lines) if OPTION.match(l))
    options = lines[first:]
    letters = [OPTION.match(o).group(1) for o in options]
    if letters != [chr(ord("A") + i) for i in range(len(options))]:
        raise ValueError(f"options are not A, B, ... in order: {options}")
    return "\n".join(lines[:first]).strip(), options


def main():
    out_images = ROOT / "data/processed/images_sparbench"
    out_images.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data/raw/SPAR-Bench/data/*.parquet"))):
        t = pd.read_parquet(f)
        for r in t[t.task.isin(MCA)].itertuples():
            stem, options = split(r.question)
            if r.answer not in [o[0] for o in options]:
                raise ValueError(f"{r.id}: answer {r.answer!r} is not an option letter")
            paths = []
            for k, im in enumerate(r.image):
                ext = (Image.open(io.BytesIO(im["bytes"])).format or "PNG").lower().replace("jpeg", "jpg")
                p = out_images / f"{r.id}_{k}.{ext}"
                p.write_bytes(im["bytes"])
                paths.append(str(p))
            rows.append(dict(source="sparbench", task=r.task, bench_source=r.source, img_type=r.img_type,
                             question_id=f"sparbench_{r.id}", images=paths, question=stem, options=options,
                             answer=r.answer, split="test"))
    rows.sort(key=lambda x: int(x["question_id"].split("_")[1]))
    with open(ROOT / "data/processed/bench_sparbench.jsonl", "w") as h:
        for x in rows:
            h.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(len(rows), "questions,", sum(len(x["images"]) for x in rows), "images")


if __name__ == "__main__":
    main()
