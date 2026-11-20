import argparse
import json
import re
from pathlib import Path

import numpy as np

MINDCUBE = "data/raw/MindCube/data/raw/MindCube_tinybench.jsonl"
RNG = np.random.default_rng(17)
B = 10000


def ci(flags):
    x = np.asarray(flags, dtype=float)
    if not len(x):
        return None
    boot = x[RNG.integers(0, len(x), (B, len(x)))].mean(1)
    return [round(float(v), 2) for v in (100 * x.mean(), 100 * np.percentile(boot, 2.5), 100 * np.percentile(boot, 97.5))]


def options(question):
    tail = question[question.rfind("?") + 1:]
    parts = re.split(r"\s*\b([A-D])\.\s*", tail)
    return {parts[i]: parts[i + 1].strip().rstrip(".").lower() for i in range(1, len(parts) - 1, 2)}


def mirror(text):
    return text.replace("left", "#").replace("right", "left").replace("#", "right")


def predictions(run):
    return {p["question_id"]: p for p in map(json.loads, open(Path(run) / "tiny/predictions.jsonl"))}


def right(p):
    return p["correct"] in (True, "True")


def two_view(run, pose):
    tiny = predictions(run)
    acc, kinds = [], []
    for r in pose:
        p, opts = tiny[r["id"]], options(r["question"])
        acc.append(right(p))
        if not right(p) and p["extracted"] in opts:
            kinds.append(opts[p["extracted"]] == mirror(opts[r["gt_answer"]]))
    return {"n": len(pose), "acc": ci(acc), "wrong": len(kinds), "invalid": acc.count(False) - len(kinds),
            "mirror_share": ci(kinds), "mirror_chance": round(100 / 3, 2)}


def state_answer(root):
    f = Path(root) / "cot_evolution_with_premise/base_axes/records.jsonl"
    recs = [r for r in map(json.loads, open(f)) if "last_direct_correct" in r]
    turned = [r["model_correct"] for r in recs if r["last_direct_correct"]]
    stalled = [r["model_correct"] for r in recs if not r["last_direct_correct"]]
    return {"turned": {"n": len(turned), "acc": ci(turned)}, "not_turned": {"n": len(stalled), "acc": ci(stalled)},
            "chance": 50.0}


def paired(a, b, pose):
    pa, pb = predictions(a), predictions(b)
    d = np.array([right(pa[r["id"]]) - right(pb[r["id"]]) for r in pose], dtype=float)
    boot = d[RNG.integers(0, len(d), (B, len(d)))].mean(1)
    return [round(float(v), 2) for v in (100 * d.mean(), 100 * np.percentile(boot, 2.5), 100 * np.percentile(boot, 97.5))]


def named(pairs):
    out = {}
    for p in pairs or []:
        name, path = p.split("=", 1)
        out[name] = path
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", help="NAME=DIR, DIR an evaluation output holding tiny/predictions.jsonl (repeat)")
    ap.add_argument("--sspace", action="append", help="NAME=DIR, a model's S-Space output directory (repeat)")
    ap.add_argument("--paired", nargs=2, action="append", metavar=("A", "B"), help="model names, A minus B (repeat)")
    ap.add_argument("--mindcube", default=MINDCUBE, help="MindCube-Tiny's raw records (with categories)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    models, sspace = named(args.model), named(args.sspace)
    pose = [r for r in map(json.loads, open(args.mindcube)) if r["category"][1] == "P-P"]
    summary = {"models": models, "two_view": {n: two_view(r, pose) for n, r in models.items()},
               "state_answer": {n: state_answer(d) for n, d in sspace.items()},
               "paired": {f"{a} - {b}": paired(models[a], models[b], pose) for a, b in args.paired or []}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=1, ensure_ascii=False)

    fmt = lambda c: "   -  " if c is None else f"{c[0]:6.2f}"
    print(f"{'':24s} {'MC pose':>8s} {'mirror%':>8s}")
    for name, t in summary["two_view"].items():
        print(f"{name:24s} {fmt(t['acc']):>8s} {fmt(t['mirror_share']):>8s}")
    print("\nSpinBench (with premise): answer accuracy by the S-Space state at the last mention")
    for name, v in summary["state_answer"].items():
        print(f"  {name:12s} in the queried frame {v['turned']['acc']} (n={v['turned']['n']})   "
              f"not {v['not_turned']['acc']} (n={v['not_turned']['n']})")
    if summary["paired"]:
        print("\npaired differences (points, 95% interval):", json.dumps(summary["paired"], indent=1))


if __name__ == "__main__":
    main()
