# Analyses

To reproduce the findings from the released checkpoints, start with the notebooks in
[`notebooks/`](notebooks/README.md); they run the scripts below and show their numbers beside the paper's.

The scripts run from the repository root. They read checkpoints, evaluation outputs (an `<eval dir>` such as
`runs/eval/qwen3.5-4b-rl/` from `scripts/eval.sh`), and, where marked, the held-out rows of the SFT dataset
(`scripts/data/sft.sh`).

## Diagnosis

**Geometry probe:** a fresh readout fit on the frozen states of probe tokens.

```bash
MODEL=<model directory> OUT=runs/probe/<name> GPU=0 bash analysis/diagnosis/geometry_probe.sh
```

**Viewpoint probe:** the probe tokens inserted after each declared viewpoint of the held-out rows.

```bash
python analysis/diagnosis/probe_view.py --ckpt runs/probe/<name>/probe/ckpt --out runs/probe/<name>/view_probe.json
```

**S-Space readouts** and the state along each model's own reasoning. The scripts use
[S-Space](https://github.com/MirroS-Lab/S-Space), cloned at `third_party/s-space` and set up with its
`scripts/install.sh` and `scripts/prepare_data.sh`. `spinbench_native_cot.py` and `cot_evolution.py` also import
`smind`, so they run with S-Space's Qwen environment. Outputs go to `runs/sspace/<name>/`; `base` is S-Space's
Qwen3.5-4B, and any other model is given with `--model_path`, a directory inside `third_party/s-space`.

```bash
python analysis/diagnosis/sspace/make_configs.py --model <name> [--model_path <model>]
GPU=0 bash analysis/diagnosis/sspace/run_queue.sh <name>
SSPY="env PYTHONPATH=third_party/s-space:src third_party/s-space/.envs/qwen36/bin/python"
$SSPY analysis/diagnosis/sspace/spinbench_native_cot.py --model <name> --premise with [--model_path <model>]
$SSPY analysis/diagnosis/sspace/cot_evolution.py replay --model <name> --premise with [--model_path <model>]
$SSPY analysis/diagnosis/sspace/cot_evolution.py score --model <name> --premise with
python analysis/diagnosis/sspace/compare.py --models base <name> ...
```

**Two-view questions** of MindCube-Tiny, and the state against the answer on SpinBench.

```bash
python analysis/diagnosis/two_view_errors.py --model <name>=<eval dir> --sspace <name>=runs/sspace/<name> --out <json>
```

## Method

**Geometry of z<sub>0</sub> and transformation of each re-emitted z<sub>k</sub>** on the held-out rows, read by the
checkpoint's readout.

```bash
python analysis/method/transformed_readout.py --ckpt <ckpt> --out <json>
```

**The readout refit at each layer** of the fusion-stage checkpoint.

```bash
python analysis/method/layer_readout.py cache --ckpt <fusion ckpt> --cache <dir>
python analysis/method/layer_readout.py probe --ckpt <fusion ckpt> --cache <dir> --layers zero mix 0 1 ... --out <json>
```

**Inference cost** on MindCube-Tiny and ViewSpatial.

```bash
python analysis/method/efficiency.py --ckpt <ckpt> --out runs/efficiency/<name>
```

**Paired difference** between two models on the four benchmarks, with bootstrap intervals.

```bash
python analysis/paired_bench.py --a <eval dir> --b <eval dir>
```
