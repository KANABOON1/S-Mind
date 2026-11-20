import json
import argparse
from pathlib import Path

from sspace_paths import LINK, SSPACE, link_outputs, model_path

BASE_AXES = "sspace/core/artifacts/pretrained/qwen35_4b_coco6000_final_logit_axes_v2"

PROJECTION = [
    "benchmarks/eval_qwen35_4b_embspatial1200_coco1800_margin.json",
    "spinbench/eval_qwen35_4b_spinbench_without_premise_rotation.json",
    "spinbench/eval_qwen35_4b_spinbench_with_premise_rotation.json",
    "multi_view/eval_qwen35_4b_hstar_hos600_multiview.json",
]
DIRECT = [
    "benchmarks/direct_qwen35_4b_embspatial1200_coco1800_margin.json",
    "spinbench/direct_qwen35_4b_spinbench_without_premise_no_thinking_greedy128.json",
    "spinbench/direct_qwen35_4b_spinbench_with_premise_no_thinking_greedy128.json",
]


def write(model, name, cfg):
    path = SSPACE / LINK / "configs" / model / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(path.relative_to(SSPACE))


def selected_layer(artifact_dir):
    return json.loads((SSPACE / artifact_dir / "manifest.json").read_text())["model"]["selected_layer"]


def emit(model, path, name, cfg, axes):
    cfg = dict(cfg, model_path=path, output_dir=f"{LINK}/{model}/{Path(name).stem}")
    if axes is not None:
        cfg["artifact_dir"] = axes
        for key in ("layer_id", "selected_layer"):
            if key in cfg:
                cfg[key] = selected_layer(axes)
    write(model, Path(name).name, cfg)


def experiments(model, path):
    for name in PROJECTION:
        emit(model, path, name, json.loads((SSPACE / "configs/experiments" / name).read_text()), BASE_AXES)
    for name in DIRECT:
        emit(model, path, name, json.loads((SSPACE / "configs/experiments" / name).read_text()), None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a name for the model's outputs (base for S-Space's Qwen3.5-4B)")
    ap.add_argument("--model_path", help="the model directory (default for base: S-Space's Qwen3.5-4B)")
    args = ap.parse_args()
    link_outputs()
    experiments(args.model, model_path(args.model, args.model_path))
