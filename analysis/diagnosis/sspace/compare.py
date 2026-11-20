import argparse
import json
from math import comb, sqrt

from sspace_paths import OUT as ROOT, SSPACE


def jsonl(path):
    return [json.loads(line) for line in path.open()]


def spinbench(variant, premise):
    d = ROOT / variant / f"eval_qwen35_4b_spinbench_{premise}_premise_rotation"
    if not (d / "summary.json").exists():
        return None
    return {r["global_sample_index"]: r["correct"]["margin"] for r in jsonl(d / "projection_records.jsonl")}


def embspatial(variant):
    d = ROOT / variant / "eval_qwen35_4b_embspatial1200_coco1800_margin"
    if not (d / "summary.json").exists():
        return None
    return {r["sample_id"]: r["correct"] for r in jsonl(d / "predictions.jsonl")}


def hstar(variant, layer=None):
    d = ROOT / variant / "eval_qwen35_4b_hstar_hos600_multiview"
    if not (d / "summary.json").exists():
        return None
    if layer is None:
        run = json.loads((d / "resolved_config.json").read_text())
        artifact = SSPACE / run["artifact_dir"]
        layer = json.loads((artifact / "manifest.json").read_text())["model"]["selected_layer"]
    records = jsonl(d / "layerwise_records.jsonl")
    i = records[0]["layer_ids"].index(layer)
    return {r["sample_index"]: r["correct_by_layer"][i] for r in records}


def direct(model, name):
    d = ROOT / model / name
    if (d / "merged").exists():
        d = d / "merged"
    if not (d / "summary.json").exists():
        return None
    return {r["global_sample_index"]: r for r in jsonl(d / "records.jsonl")}


def native_cot(model, premise):
    path = ROOT / model / f"native_cot_spinbench_{premise}" / "summary.json"
    if not path.exists():
        return None
    return {r["sample_id"]: {"correct": r["correct"], "parse_failure": r["pred"] is None,
                             "ended_with_eos": not r["truncated"]} for r in jsonl(path.with_name("predictions.jsonl"))}


def paired(ref, new):
    if ref.keys() != new.keys():
        raise ValueError("paired comparison over different question sets")
    n = len(ref)
    b = sum(new[k] and not ref[k] for k in ref)
    c = sum(ref[k] and not new[k] for k in ref)
    d = (b - c) / n
    se = sqrt(b + c - (b - c) ** 2 / n) / n
    p = min(1.0, 2 * sum(comb(b + c, i) for i in range(min(b, c) + 1)) / 2 ** (b + c)) if b + c else 1.0
    return {"diff": d, "ci95": [d - 1.96 * se, d + 1.96 * se], "only_new": b, "only_base": c, "mcnemar_p": p}


def row(results):
    out = {}
    records = {v: r for v, r in results.items() if r is not None and isinstance(next(iter(r.values())), dict)}
    results = {v: ({k: x["correct"] for k, x in r.items()} if v in records else r)
               for v, r in results.items() if r is not None}
    ref = results.get("base")
    for v, r in results.items():
        out[v] = {"n": len(r), "accuracy": sum(r.values()) / len(r)}
        if v in records:
            out[v]["invalid_rate"] = sum(bool(x["parse_failure"]) for x in records[v].values()) / len(r)
            out[v]["truncation_rate"] = sum(not x["ended_with_eos"] for x in records[v].values()) / len(r)
        if v != "base" and ref is not None:
            out[v]["vs_base"] = paired(ref, r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["base"], help="the names given to make_configs.py; base first")
    VARIANTS = ap.parse_args().models
    report = {
        "projection": {
            "spinbench_with_premise_rotation": row({v: spinbench(v, "with") for v in VARIANTS}),
            "spinbench_without_premise_rotation": row({v: spinbench(v, "without") for v in VARIANTS}),
            "embspatial_margin": row({v: embspatial(v) for v in VARIANTS}),
            "hstar_multiview_selected_layer": row({v: hstar(v) for v in VARIANTS}),
        },
        "direct": {
            name: row({m: direct(m, name) for m in VARIANTS})
            for name in ["direct_qwen35_4b_embspatial1200_coco1800_margin",
                         "direct_qwen35_4b_spinbench_with_premise_no_thinking_greedy128",
                         "direct_qwen35_4b_spinbench_without_premise_no_thinking_greedy128"]
        },
        "native_cot": {
            f"spinbench_{premise}_premise": row({m: native_cot(m, premise) for m in VARIANTS})
            for premise in ["with", "without"]
        },
    }
    (ROOT / "comparison.json").write_text(json.dumps(report, indent=1) + "\n")
    for section in ("projection", "direct", "native_cot"):
        for name, rows in report[section].items():
            print(name)
            for v, r in rows.items():
                vs = r.get("vs_base")
                extra = (f"  diff {vs['diff']:+.4f} [{vs['ci95'][0]:+.4f}, {vs['ci95'][1]:+.4f}] "
                         f"{vs['only_new']}/{vs['only_base']} p={vs['mcnemar_p']:.3g}") if vs else ""
                rates = f" invalid={r['invalid_rate']:.4f} trunc={r['truncation_rate']:.4f}" if "invalid_rate" in r else ""
                print(f"  {v:14s} n={r['n']:4d} acc={r['accuracy']:.4f}{rates}{extra}")


if __name__ == "__main__":
    main()
