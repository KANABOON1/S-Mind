import argparse
import json

import numpy as np
import torch
from safetensors.numpy import load_file, save_file

from sspace.core.artifacts import SSpaceArtifact
from sspace.core.extraction.execution import move_model_inputs
from sspace.core.models.runtime import configure_image_max_pixels, load_model_runtime
from sspace.determinism import configure_deterministic_cuda
from sspace.experiments.spinbench.evolving_cot.runtime import _append_generated, _mentions
from sspace.experiments.spinbench.perspective_taking.adapter import (
    load_spinbench,
    rotate_coordinates,
    spinbench_target_margin,
)
from sspace.experiments.spinbench.perspective_taking.direct import _load_image
from sspace_paths import OUT, SSPACE, model_path

BASE_AXES = SSPACE / "sspace/core/artifacts/pretrained/qwen35_4b_coco6000_final_logit_axes_v2"
PREMISE = "without"


def evolution_dir(premise):
    return "cot_evolution" if premise == PREMISE else f"cot_evolution_{premise}_premise"


def _samples(premise):
    official = json.loads((SSPACE / "configs/experiments/spinbench"
                           / f"direct_qwen35_4b_spinbench_{premise}_premise_no_thinking_greedy128.json").read_text())
    return official, {s.sample_id: s for s in load_spinbench(
        SSPACE / official["annotation_path"], SSPACE / official["image_root"], premise)}


def _answer_start(ids, tokenizer):
    cut = tokenizer.decode(ids, skip_special_tokens=True).find("<answer>")
    if cut < 0:
        return len(ids)
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tokenizer.decode(ids[:mid], skip_special_tokens=True)) <= cut:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _reasoning_mentions(ids, tokenizer, sample):
    stop = _answer_start(ids, tokenizer)
    found = _mentions(ids[:stop], tokenizer, {"object_a": sample.object_a, "object_b": sample.object_b})
    kept = [m for m in found if not any(
        o["role"] != m["role"] and o["generated_token_start"] <= m["generated_token_start"]
        and m["generated_token_end"] <= o["generated_token_end"]
        and (o["generated_token_end"] - o["generated_token_start"]) > (m["generated_token_end"] - m["generated_token_start"])
        for o in found)]
    by_end = {}
    for m in kept:
        by_end.setdefault((m["role"], m["generated_token_end"] - 1), m)
    return stop, sorted(by_end.values(), key=lambda m: m["generated_token_end"])


def replay_states(out, preds, samples, runtime, prompt_inputs, append=_append_generated):
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = runtime.processor.tokenizer
    meta_path = out / "mentions.jsonl"
    done = {json.loads(line)["sample_id"] for line in meta_path.open()} if meta_path.exists() else set()
    with meta_path.open("a") as meta:
        for pred in preds:
            if pred["sample_id"] in done:
                continue
            sample = samples[pred["sample_id"]]
            ids = pred["generated_ids"]
            stop, mentions = _reasoning_mentions(ids, tokenizer, sample)
            row = {"sample_id": sample.sample_id, "reasoning_tokens": stop, "mentions": mentions}
            if mentions:
                inputs = prompt_inputs(sample, pred)
                n_prompt = int(inputs["input_ids"].shape[1])
                append(inputs, ids)
                positions = [n_prompt + m["generated_token_end"] - 1 for m in mentions]
                captured = {}
                hooks = [block.register_forward_hook(
                    lambda _m, _i, output, layer=layer: captured.__setitem__(
                        layer, (output[0] if isinstance(output, tuple) else output)[0, positions].float().cpu()))
                    for layer, block in enumerate(runtime.blocks)]
                try:
                    with torch.inference_mode():
                        runtime.model(**move_model_inputs(inputs, runtime.model), use_cache=False, logits_to_keep=1)
                finally:
                    for hook in hooks:
                        hook.remove()
                states = torch.stack([captured[layer] for layer in range(len(runtime.blocks))])
                if not torch.isfinite(states).all():
                    raise RuntimeError(f"non-finite states in {sample.sample_id}")
                save_file({"states": states.numpy()}, out / f"{sample.sample_id}.safetensors")
            meta.write(json.dumps(row) + "\n")
            meta.flush()
    print(f"replayed {len(preds)} samples -> {out}")


def replay(model, path, premise):
    official, samples = _samples(premise)
    preds = [json.loads(line) for line in (OUT / model / f"native_cot_spinbench_{premise}/predictions.jsonl").open()]
    if len(preds) != len(samples):
        raise ValueError(f"{len(preds)} native-CoT predictions for {len(samples)} samples")
    configure_deterministic_cuda()
    torch.manual_seed(official["seed"])
    runtime = load_model_runtime("qwen35_4b", SSPACE / path, "cuda:0")
    configure_image_max_pixels(runtime, official["image_max_pixels"])
    replay_states(OUT / model / evolution_dir(premise) / "states", preds, samples, runtime,
                  lambda sample, pred: runtime.prepare_generation(runtime.processor, _load_image(sample)[0], pred["prompt"]))


def _readout(pair, sample, mapping):
    v = pair if mapping == "direct" else rotate_coordinates(pair, sample.target_view)
    score = float(spinbench_target_margin(v, sample.target_property))
    return "A" if score > 0 else "B" if score < 0 else None


def score(model, axes_name, premise):
    artifact = SSpaceArtifact.load(BASE_AXES)
    _, samples = _samples(premise)
    preds = {json.loads(l)["sample_id"]: json.loads(l)
             for l in (OUT / model / f"native_cot_spinbench_{premise}/predictions.jsonl").open()}
    score_states(OUT / model / evolution_dir(premise), preds, samples, artifact, model, axes_name, premise)


def score_states(root, preds, samples, artifact, model, axes_name, premise):
    layer = artifact.manifest.model.selected_layer
    layers = list(artifact.manifest.model.layer_ids)
    states_dir = root / "states"
    meta = [json.loads(line) for line in (states_dir / "mentions.jsonl").open()]
    if len(meta) != len(samples):
        raise ValueError("replay incomplete")

    rows = []
    for m in meta:
        sample, pred = samples[m["sample_id"]], preds[m["sample_id"]]
        row = {"sample_id": m["sample_id"], "target_view": sample.target_view,
               "target_property": sample.target_property, "gold": sample.answer, "model_pred": pred["pred"],
               "model_correct": pred["correct"],
               "n_mentions_a": sum(x["role"] == "object_a" for x in m["mentions"]),
               "n_mentions_b": sum(x["role"] == "object_b" for x in m["mentions"])}
        if m["mentions"]:
            states = load_file(states_dir / f"{m['sample_id']}.safetensors")["states"].astype(np.float64)
            cols = {"object_a": [], "object_b": []}
            for i, x in enumerate(m["mentions"]):
                cols[x["role"]].append(i)
            stages = {}
            if len(cols["object_a"]) >= 2 and len(cols["object_b"]) >= 2:
                stages["first"] = (cols["object_a"][0], cols["object_b"][0])
                stages["last"] = (cols["object_a"][-1], cols["object_b"][-1])
            n = m["reasoning_tokens"]
            for k, name in enumerate(("early", "middle", "late")):
                inside = {r: [i for i in cols[r] if k * n <= 3 * (m["mentions"][i]["generated_token_end"] - 1) < (k + 1) * n]
                          for r in cols}
                if inside["object_a"] and inside["object_b"]:
                    stages[name] = (inside["object_a"][-1], inside["object_b"][-1])
            for name, (ia, ib) in stages.items():
                for li in layers:
                    pair = (states[li, ia] - states[li, ib]) @ artifact.axes(li).T.astype(np.float64)
                    for mapping in ("direct", "rotated"):
                        ans = _readout(pair, sample, mapping)
                        key = f"{name}_{mapping}" if li == layer else f"L{li}_{name}_{mapping}"
                        row[f"{key}_answer"] = ans
                        row[f"{key}_correct"] = ans == sample.answer
        rows.append(row)

    def acc(key, subset=None):
        sub = [r for r in (subset or rows) if f"{key}_correct" in r]
        return {"n": len(sub), "accuracy": (sum(r[f"{key}_correct"] for r in sub) / len(sub)) if sub else None}

    def paired(a, b):
        sub = [r for r in rows if f"{a}_correct" in r and f"{b}_correct" in r]
        return {"n": len(sub), "only_" + a: sum(r[f"{a}_correct"] and not r[f"{b}_correct"] for r in sub),
                "only_" + b: sum(r[f"{b}_correct"] and not r[f"{a}_correct"] for r in sub)}

    stages = ("first", "last", "early", "middle", "late")
    wrong = [r for r in rows if not r["model_correct"]]
    ended = [r for r in rows if "last_direct_answer" in r]
    summary = {
        "model": model, "axes": axes_name, "artifact_id": artifact.manifest.artifact_id, "selected_layer": layer,
        "premise": premise, "n": len(rows),
        "model_accuracy": sum(r["model_correct"] for r in rows) / len(rows),
        "both_objects_mentioned": sum(r["n_mentions_a"] > 0 and r["n_mentions_b"] > 0 for r in rows),
        "both_mentioned_twice": sum(r["n_mentions_a"] >= 2 and r["n_mentions_b"] >= 2 for r in rows),
        "stages": {s: {mp: acc(f"{s}_{mp}") for mp in ("direct", "rotated")} for s in stages},
        "paired": {"last_direct_vs_first_direct": paired("last_direct", "first_direct"),
                   "first_rotated_vs_last_rotated": paired("first_rotated", "last_rotated")},
        "last_direct_agrees_with_model_answer": {"n": len(ended), "rate": (
            sum(r["last_direct_answer"] == r["model_pred"] for r in ended) / len(ended)) if ended else None},
        "on_model_wrong": {f"{s}_{mp}": acc(f"{s}_{mp}", wrong) for s in ("first", "last") for mp in ("direct", "rotated")},
        "layerwise": {f"L{li}": {f"{s}_{mp}": acc((f"L{li}_" if li != layer else "") + f"{s}_{mp}")["accuracy"]
                                 for s in ("first", "last") for mp in ("direct", "rotated")} for li in layers},
    }
    out = root / f"{axes_name}_axes"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "records.jsonl").open("w") as stream:
        for r in rows:
            stream.write(json.dumps(r) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("model", "axes", "selected_layer", "model_accuracy",
                                              "both_mentioned_twice", "stages")}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["replay", "score"])
    parser.add_argument("--model", required=True, help="the name used with spinbench_native_cot.py")
    parser.add_argument("--model_path", help="replay: the model directory (default for base: S-Space's Qwen3.5-4B)")
    parser.add_argument("--premise", choices=["with", "without"], default=PREMISE,
                        help="the SpinBench variant; with premise goes to <model>/cot_evolution_with_premise/")
    args = parser.parse_args()
    if args.step == "replay":
        replay(args.model, model_path(args.model, args.model_path), args.premise)
    else:
        score(args.model, "base", args.premise)
