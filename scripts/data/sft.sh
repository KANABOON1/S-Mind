#!/usr/bin/env bash
main() {
set -euo pipefail
cd "$(dirname "$0")/../.."
D=data/smind/sft/gpt-6-astra
hf download --repo-type dataset KANABOON1/S-Mind-Data data.jsonl --local-dir "$D"
python -m smind.data.sft.build --dataset "$D/data.jsonl"
python -m smind.data.sft.rows --out "$D/sft_2500_sview"
python -m smind.data.scene_group --src "$D/sft_2500_sview" --dst "$D/perspective" --m 2 --c 4
}
main "$@"; exit $?
