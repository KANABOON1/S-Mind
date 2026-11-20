# Reproducing the analyses

These notebooks reproduce the paper's findings about S-Mind and about its base model, Qwen3.5, from the released
checkpoints. Each one runs the scripts of [`analysis/`](../../analysis/README.md), skips every step whose output already
exists under `runs/notebooks/`, and shows its numbers beside the paper's. They download the checkpoints they need from
[KANABOON1/S-Mind](https://huggingface.co/KANABOON1/S-Mind) into `models/`.

Set `SIZE` and `GPU` in the first cell of each notebook. With `SIZE = "4b"` the paper's numbers are shown beside the
results; `SIZE = "9b"` runs the same analyses on S-Mind-9B.

| Notebook | Paper | Data ([scripts](../../README.md#data)) | Time on one H100 |
|---|---|---|---|
| [`01_accuracy_and_relating_views`](01_accuracy_and_relating_views.ipynb) | Table 1 (S-Mind rows), Fig. 8b | `benchmarks.sh` | about 1 h per model |
| [`02_scene_in_the_tokens`](02_scene_in_the_tokens.ipynb) | Fig. 8c, App. D.4–D.5 | `geometry.sh`, `sft.sh` | about 5 min per model |
| [`03_views_fused`](03_views_fused.ipynb) | Fig. 3a, Fig. 6a, Table 3 | `geometry.sh` | about 3 h per model (4 h for 9B) |
| [`04_scene_turns_with_the_reasoning`](04_scene_turns_with_the_reasoning.ipynb) | Fig. 3b, Fig. 6b, Table 3 | `sft.sh`, and the probes of `03` | about 1 h per model |
| [`05_middle_layers`](05_middle_layers.ipynb) | Fig. 6c | `geometry.sh` | 15 min on the GPU, then about 20 min of CPU fits |
| [`06_base_diagnosis`](06_base_diagnosis.ipynb) | Fig. 3b–c, App. A (Table 4) | `benchmarks.sh`, and [S-Space](https://github.com/MirroS-Lab/S-Space) | about 1.5 h |

Run the notebooks from this folder; `scripts/setup.sh` installs the packages they use.
