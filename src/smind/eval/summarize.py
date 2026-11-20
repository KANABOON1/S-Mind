import argparse
import json
from pathlib import Path

from smind.eval.sets import BENCHES, SETS, summary


def summarize(out, sets):
    for name in sets:
        d = Path(out) / name
        rows = [json.loads(l) for p in sorted(d.glob("predictions_shard*.jsonl")) for l in open(p)]
        with open(d / "predictions.jsonl", "w") as f:
            for r in sorted(rows, key=lambda r: r["question_id"]):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        s = summary(rows)
        (d / "summary.json").write_text(json.dumps(s, indent=1))
        print(name, json.dumps({k: v for k, v in s.items() if k != "per_task"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the evaluation output directory (generate_ours --out)")
    ap.add_argument("--sets", nargs="+", default=BENCHES,
                    help=f"set directories under --out: names in {sorted(SETS)}")
    args = ap.parse_args()
    summarize(args.out, args.sets)


if __name__ == "__main__":
    main()
