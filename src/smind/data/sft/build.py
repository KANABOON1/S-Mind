import argparse
import collections
import json
import random
from pathlib import Path

import pandas as pd

from smind import formats
from smind.data.geometry.maps import K, MAPS_DIR, insert_scene
from smind.model.layout import Z_SLOT, Z_TRIGGER
from smind.model.readout import load_maps

SFT_DIR = Path("data/smind/sft")
SOURCES = ("mindcube", "spar", "spatialladder")
Z = "[Z]"  # where the released chains mark the spatial tokens


def select(responses, per_source, seed):
    rng = random.Random(seed)
    picked, report = {}, {}
    for source in SOURCES:
        recs = [json.loads(line) for line in open(f"data/processed/{source}_train.jsonl")]
        passing = [r for r in recs if r["question_id"] in responses]
        if len(passing) < per_source:
            raise ValueError(f"{source}: only {len(passing)} questions with a passing response")
        picked[source] = rng.sample(passing, per_source)
        report[source] = dict(total=per_source, tasks=dict(collections.Counter(r["task"] for r in picked[source])))
    return picked, report


def write_sft(recs, data, by_images, prefix):
    def rows(user, answer):
        return pd.DataFrame([dict(messages=[dict(role="user", content=user(r)), dict(role="assistant", content=answer(r))],
                                  images=[dict(image=p) for p in r["images"]], enable_thinking=False,
                                  question_id=r["question_id"], source=r["source"], task=r["task"]) for r in recs])
    scene = lambda r: insert_scene(formats.user_text(r))
    target = lambda r: data[r["question_id"]]["marked"].strip().replace(" " + Z, Z_TRIGGER + Z_SLOT * K)
    rows(scene, target).to_parquet(f"{prefix}_z.parquet")
    with open(f"{prefix}_maps.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(dict(by_images[tuple(r["images"])], question_id=r["question_id"])) + "\n")
    # the viewpoint each spatial-token run is re-expressed at, row by row: the targets of the view loss
    Path(f"{prefix}_views.json").write_text(json.dumps([data[r["question_id"]]["views"] for r in recs]))
    print(prefix, len(recs))


def sft(args):
    data = {}
    for line in open(args.dataset):
        row = json.loads(line)
        if len(row["views"]) != row["marked"].count(Z):
            raise ValueError(f"{row['question_id']}: {len(row['views'])} views for {row['marked'].count(Z)} [Z]")
        data[row["question_id"]] = row
    by_images = {tuple(m["images"]): m for path in (MAPS_DIR / "maps_train.jsonl", MAPS_DIR / "maps_scannet_train.jsonl")
                 for m in load_maps(path)}
    picked, report = select(data, args.per_source, args.seed)
    prefix = Path(args.dataset).parent / f"sft_{args.per_source}"
    write_sft([r for s in SOURCES for r in picked[s]], data, by_images, prefix)
    Path(f"{prefix}.report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="the released data.jsonl (S-Mind-Data); the set is written beside it")
    ap.add_argument("--per_source", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=17)
    sft(ap.parse_args())


if __name__ == "__main__":
    main()
