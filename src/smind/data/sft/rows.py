import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from smind.data.geometry.maps import FLAT_TOKENS as SCENE_TOKENS, K
from smind.model.layout import Z_SLOT

ROOT = Path(__file__).resolve().parents[4]
SRC = Path("data/smind/sft/gpt-6-astra")
HELDOUT = ROOT / "configs/splits/sft_heldout.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC), help="the folder of sft_2500_z.parquet, sft_2500_maps.jsonl and "
                                                   "sft_2500_views.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    df = pd.read_parquet(src / "sft_2500_z.parquet")
    maps = [json.loads(l) for l in open(src / "sft_2500_maps.jsonl")]
    actions = json.loads((src / "sft_2500_views.json").read_text())
    if not len(df) == len(maps) == len(actions) or any(m["question_id"] != q for m, q in zip(maps, df.question_id)):
        raise ValueError("rows, maps and views out of step")
    slots, scene = Z_SLOT * K, "".join(SCENE_TOKENS)
    messages = []
    for (_, row), acts in zip(df.iterrows(), actions):
        user, answer = row["messages"]
        text = answer["content"].replace("<|z|>", "")
        if text.count(slots) != len(acts) or scene in text:
            raise ValueError(f"{row['question_id']}: {text.count(slots)} slot runs, {len(acts)} views")
        messages.append([user, dict(answer, content=text.replace(slots, scene))])
    df = df.assign(messages=messages)
    held = set(json.loads(HELDOUT.read_text()))
    if not held <= set(df.question_id):
        raise ValueError(f"{len(held - set(df.question_id))} held-out rows of {HELDOUT} are not in the dataset")
    for split, keep in (("train", ~df.question_id.isin(held)), ("heldout", df.question_id.isin(held))):
        idx = np.flatnonzero(keep.to_numpy())
        df.iloc[idx].reset_index(drop=True).to_parquet(out / f"{split}_z.parquet")
        with open(out / f"{split}_maps.jsonl", "w") as f:
            for i in idx:
                f.write(json.dumps(maps[i]) + "\n")
        (out / f"{split}_targets.json").write_text(json.dumps([actions[i] for i in idx]))
        print(split, len(idx), "rows", dict(df.source.iloc[idx].value_counts()), sum(len(actions[i]) for i in idx),
              "spatial-token runs")


if __name__ == "__main__":
    main()
