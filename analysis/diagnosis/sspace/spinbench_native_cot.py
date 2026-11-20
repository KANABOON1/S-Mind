import argparse
import json
import random

import numpy as np
import torch

from sspace.core.models.runtime import configure_image_max_pixels, load_model_runtime
from sspace.determinism import configure_deterministic_cuda
from sspace.core.extraction.execution import move_model_inputs
from sspace.experiments.spinbench.perspective_taking.adapter import load_spinbench, spinbench_fingerprint
from sspace.experiments.spinbench.perspective_taking.direct import _eos_ids, _load_image
from smind.formats import SUFFIX, extract_answer
from sspace_paths import OUT, SSPACE, model_path

OFFICIAL_INSTRUCTION = "Only answer with a single capital letter from (A, B)."
MAX_NEW_TOKENS = 4096


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="a name for the outputs (base for S-Space's Qwen3.5-4B)")
    parser.add_argument("--model_path", help="the model directory (default for base: S-Space's Qwen3.5-4B)")
    parser.add_argument("--premise", choices=["with", "without"], required=True)
    args = parser.parse_args()

    official = json.loads((SSPACE / "configs/experiments/spinbench"
                           / f"direct_qwen35_4b_spinbench_{args.premise}_premise_no_thinking_greedy128.json").read_text())
    samples = load_spinbench(SSPACE / official["annotation_path"], SSPACE / official["image_root"], args.premise)
    if spinbench_fingerprint(samples) != official["expected_dataset_fingerprint"]:
        raise ValueError("SpinBench dataset fingerprint changed")

    out = OUT / args.model / f"native_cot_spinbench_{args.premise}"
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"
    done = {json.loads(line)["sample_id"] for line in pred_path.open()} if pred_path.exists() else set()

    configure_deterministic_cuda()
    random.seed(official["seed"]); np.random.seed(official["seed"]); torch.manual_seed(official["seed"])
    torch.use_deterministic_algorithms(True)
    runtime = load_model_runtime("qwen35_4b", SSPACE / model_path(args.model, args.model_path), "cuda:0")
    configure_image_max_pixels(runtime, official["image_max_pixels"])
    tokenizer = runtime.processor.tokenizer
    eos_ids = _eos_ids(runtime.model)

    with pred_path.open("a") as stream:
        for sample in samples:
            if sample.sample_id in done:
                continue
            lines = sample.direct_prompt.splitlines()
            if lines[-1] != OFFICIAL_INSTRUCTION:
                raise ValueError(f"unexpected SpinBench instruction in {sample.sample_id}")
            prompt = "\n".join(lines[:-1] + [SUFFIX])
            image, _ = _load_image(sample)
            inputs = move_model_inputs(runtime.prepare_generation(runtime.processor, image, prompt), runtime.model)
            n_in = int(inputs["input_ids"].shape[1])
            with torch.inference_mode():
                output = runtime.model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS)
            generated = output[0, n_in:].tolist()
            text = tokenizer.decode(generated, skip_special_tokens=True)
            pred = extract_answer(text, 2)
            stream.write(json.dumps({
                "sample_id": sample.sample_id, "transform": sample.transform, "target_view": sample.target_view,
                "target_property": sample.target_property, "gold": sample.answer, "pred": pred,
                "correct": pred == sample.answer, "truncated": generated[-1] not in eos_ids,
                "output_tokens": len(generated), "prompt": prompt, "response": text,
                "generated_ids": generated,
            }) + "\n")
            stream.flush()

    records = [json.loads(line) for line in pred_path.open()]
    if len(records) != len(samples):
        raise ValueError(f"{len(records)} predictions for {len(samples)} samples")
    n = len(records)
    summary = {
        "model": args.model, "premise": args.premise, "n": n, "max_new_tokens": MAX_NEW_TOKENS,
        "accuracy": sum(r["correct"] for r in records) / n,
        "invalid_rate": sum(r["pred"] is None for r in records) / n,
        "truncation_rate": sum(r["truncated"] for r in records) / n,
        "mean_output_tokens": sum(r["output_tokens"] for r in records) / n,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
