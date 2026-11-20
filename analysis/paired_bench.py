import argparse
import ast
import json
import random
import re
from collections import defaultdict

from smind.eval.sets import BENCHES, set_path



def correct(run, bench):
    return {r["question_id"]: str(r["correct"]) == "True"
            for r in map(json.loads, open(f"{run}/{bench}/predictions.jsonl"))}


def unit(bench, qid):
    m = re.match(r"^([a-z]+_group\d+)", qid) if bench == "tiny" else None
    return m.group(1) if m else qid


def score(bench, rows, task):
    if bench == "sparbench":
        by = defaultdict(lambda: [0, 0, 0])
        for q, a, b in rows:
            t = by[task[q]]
            t[0] += a; t[1] += b; t[2] += 1
        return tuple(100 * sum(v[i] / v[2] for v in by.values()) / len(by) for i in (0, 1))
    return tuple(100 * sum(r[i] for r in rows) / len(rows) for i in (1, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="an evaluation output directory with <bench>/predictions.jsonl")
    ap.add_argument("--b", required=True)
    ap.add_argument("--sets", nargs="+", default=BENCHES, choices=BENCHES)
    ap.add_argument("--images", type=int, help="only the questions with this many input images")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    units, tasks, point, boot = {}, {}, {}, defaultdict(list)
    for bench in args.sets:
        meta = {}
        for r in map(json.loads, open(set_path(bench))):
            images = ast.literal_eval(r["images"]) if isinstance(r["images"], str) else r["images"]
            meta[r["question_id"]] = (r["task"], len(images))
        a, b = correct(args.a, bench), correct(args.b, bench)
        if set(a) != set(b) or set(a) != set(meta):
            raise ValueError(f"{bench}: {len(a)} / {len(b)} answered of {len(meta)} questions")
        tasks[bench] = {q: t for q, (t, _) in meta.items()}
        strata = defaultdict(lambda: defaultdict(list))
        for q, (t, n) in meta.items():
            if args.images is None or n == args.images:
                strata[t][unit(bench, q)].append((q, a[q], b[q]))
        units[bench] = [list(groups.values()) for groups in strata.values()]
        point[bench] = score(bench, [x for s in units[bench] for g in s for x in g], tasks[bench])
    for _ in range(args.n):
        for bench in args.sets:
            draw = [x for s in units[bench] for _ in s for x in s[rng.randrange(len(s))]]
            x, y = score(bench, draw, tasks[bench])
            boot[bench].append(x - y)
    rows = list(args.sets)
    if args.sets == BENCHES and args.images is None:
        point["avg"] = tuple(sum(point[b][i] for b in BENCHES) / 4 for i in (0, 1))
        boot["avg"] = [sum(v) / 4 for v in zip(*(boot[b] for b in BENCHES))]
        rows.append("avg")
    out = {}
    for name in rows:
        s = sorted(boot[name])
        p = min(1.0, 2 * min(sum(v <= 0 for v in s), sum(v >= 0 for v in s)) / args.n)
        x, y = point[name]
        out[name] = dict(a=x, b=y, diff=x - y, ci=[s[int(0.025 * args.n)], s[int(0.975 * args.n) - 1]], p=p,
                         n=sum(len(g) for st in units[name] for g in st) if name in units else None)
        print(f"{name:13s} {x:6.2f} vs {y:6.2f}  {x - y:+6.2f} [{out[name]['ci'][0]:+.2f}, {out[name]['ci'][1]:+.2f}]  "
              f"p={p:.3f}" + (f"  n={out[name]['n']}" if out[name]["n"] else ""))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(a=args.a, b=args.b, images=args.images, rows=out), f, indent=1)


if __name__ == "__main__":
    main()
