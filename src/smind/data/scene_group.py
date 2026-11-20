import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

FLAT = re.compile("".join(re.escape(f"<|scene_{i}|>") for i in range(16)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--m", type=int, required=True, help="tokens per image")
    ap.add_argument("--c", type=int, required=True, help="camera tokens")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=False)
    group = "".join(f"<|scene_{i}|>" for i in range(args.m))
    cams = "".join(f"<|cam_{i}|>" for i in range(args.c))
    for f in sorted(src.iterdir()):
        if f.is_dir():
            continue
        if f.suffix != ".parquet":
            shutil.copy(f, dst / f.name)
            continue
        t = pd.read_parquet(f)
        rows, runs, groups = [], 0, 0
        for ms in t.messages:
            n = sum(m["content"].count("<image>") for m in ms if m["role"] == "user")
            if n == 0:
                raise ValueError(f"{f.name}: a row without images")
            k = sum(len(FLAT.findall(m["content"])) for m in ms)
            runs, groups = runs + k, groups + k * n
            rows.append([dict(m, content=FLAT.sub(group * n + cams, m["content"])) for m in ms])
        text = "".join(m["content"] for ms in rows for m in ms)
        if (max((int(x) for x in re.findall(r"<\|scene_(\d+)\|>", text)), default=0) >= args.m
                or text.count(cams) != runs or text.count("<|scene_0|>") != groups):
            raise ValueError(f"{f.name}: {runs} flat runs did not become {groups} image groups and {runs} camera groups")
        t["messages"] = rows
        t.to_parquet(dst / f.name)
        print(f.name, len(t), "rows,", runs, "runs ->", f"groups of {args.m} per image + {args.c} cameras")


if __name__ == "__main__":
    main()
