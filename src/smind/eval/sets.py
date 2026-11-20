import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BENCHES = ["tiny", "sparbench", "blink_spatial", "viewspatial"]
DEV = ["dev_mindcube", "dev_spar", "dev_spatialladder"]
SETS = {
    "tiny": "data/processed/mindcube_tiny.jsonl",
    "sparbench": "data/processed/bench_sparbench.jsonl",
    "blink_spatial": "data/processed/bench_blink_spatial.jsonl",
    "viewspatial": "data/processed/bench_viewspatial.jsonl",
    "dev_mindcube": "data/processed/mindcube_dev.jsonl",
    "dev_spar": "data/processed/spar_dev.jsonl",
    "dev_spatialladder": "data/processed/spatialladder_dev.jsonl",
}


def set_path(name):
    return ROOT / SETS[name] if name in SETS else Path(name)


def records(name):
    return [json.loads(line) for line in open(set_path(name))]


def summary(rows):
    n = len(rows)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    acc = lambda rs: sum(bool(r["correct"]) for r in rs) / len(rs)
    out = dict(n=n, acc=acc(rows), invalid=sum(r["extracted"] is None for r in rows) / n,
               truncated=sum(r["finish_reason"] == "length" for r in rows) / n)
    if all("triggers" in r for r in rows):
        out["mean_triggers"] = sum(r["triggers"] for r in rows) / n
    out["per_task"] = {t: dict(n=len(rs), acc=acc(rs)) for t, rs in sorted(by.items())}
    sources = {r.get("source") for r in rows}
    if sources == {"sparbench"}:
        out["acc_task_mean"] = sum(v["acc"] for v in out["per_task"].values()) / len(out["per_task"])
    if sources == {"viewspatial"}:
        for group in ("Camera", "Person"):
            rs = [r for r in rows if r["task"].startswith(f"{group} perspective")]
            if rs:  # a subset of the questions may miss a group
                out[f"acc_{group.lower()}"] = acc(rs)
    return out
