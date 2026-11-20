#!/usr/bin/env bash
main() {
set -euo pipefail
cd "$(dirname "$0")/../.."
for d in MLL-Lab/MindCube:MindCube hongxingli/SpatialLadder-26k:SpatialLadder-26k jasonzhango/SPAR-Bench:SPAR-Bench \
         BLINK-Benchmark/BLINK:BLINK lidingm/ViewSpatial-Bench:ViewSpatial-Bench; do
  hf download --repo-type dataset "${d%%:*}" --local-dir "data/raw/${d##*:}"
done
[ -d data/raw/MindCube/data/raw ] || unzip -q data/raw/MindCube/data.zip -d data/raw/MindCube
mkdir -p third_party/MindCube/data/raw
cp data/raw/MindCube/data/raw/MindCube_train.jsonl third_party/MindCube/data/raw/
(cd third_party/MindCube && python scripts/data_processing.py --input data/raw/MindCube_train.jsonl --task full_pipeline)
python -m smind.data.build
python -m smind.data.build_sparbench
python -m smind.data.build_blink
python -m smind.data.build_viewspatial
}
main "$@"; exit $?
