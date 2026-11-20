#!/usr/bin/env bash
main() {
set -euo pipefail
cd "$(dirname "$0")/../.."
SCANNET_URL=${SCANNET_URL:?the download URL you receive with ScanNet access}
[ -d data/raw/ScanNet/frames_25k/scannet_frames_25k ] || {
  echo "unzip ScanNet's scannet_frames_25k into data/raw/ScanNet/frames_25k/ first" >&2; exit 1; }
hf download --repo-type dataset jasonzhango/SPAR-7M --include scannet.tar.gz --local-dir data/raw/SPAR-7M
[ -d data/raw/SPAR-7M/scannet ] || tar -xzf data/raw/SPAR-7M/scannet.tar.gz -C data/raw/SPAR-7M
python -m smind.data.build_spar select
python -m smind.data.scannet_fetch --frames data/raw/ScanNet/spar_frames.json --base_url "$SCANNET_URL"
python -m smind.data.scannet_fetch --frames data/raw/ScanNet/sl_frames.json --base_url "$SCANNET_URL"
python -m smind.data.build_spar build
python -m smind.data.geometry.maps
python -m smind.data.geometry.scannet_map
python -m smind.data.geometry.scannet_map --token_points
python -m smind.data.geometry.mindcube_points
python -m smind.data.geometry.scannet_regroup --out data/smind/maps_x
python -m smind.data.scene_group --src data/smind/maps_x --dst data/smind/fusion --m 2 --c 4
python -m smind.data.scene_group --src data/smind/maps --dst data/smind/probe --m 2 --c 4
}
main "$@"; exit $?
