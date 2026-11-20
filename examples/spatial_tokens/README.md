# Decoding the spatial tokens

S-Mind holds the scene of its input images in a few spatial tokens: z<sub>0</sub> after the images, and a new z<sub>k</sub>
each time its reasoning declares a viewpoint. This example answers one question with a released checkpoint and decodes
every z with the checkpoint's readout:

- `viewer.html`: an interactive 3D view of every z, in the style of the project page.
- `topdown.png`: a top-down view of each z. z<sub>0</sub> is in the frame of the first image, with the decoded cameras;
  each z<sub>k</sub> is the scene seen from the viewpoint the reasoning declared just before it (the viewer at the origin,
  facing up). Points take the colors of the photos.
- `z0.ply`, `z1.ply`, ...: the decoded points, colored by the image patch they come from.
- `result.json`: the reasoning, the answer, the declared viewpoints and the decoded cameras.

The eight scenes of `gallery.jsonl` (SPAR-Bench and the SPAR development split) are the ones shown on the project page.
They need SPAR-Bench and the SPAR split built (`scripts/data/benchmarks.sh` and `scripts/data/geometry.sh`). From the repository
root:

```bash
hf download KANABOON1/S-Mind --include "qwen3.5-4b/rl/*" --local-dir models/S-Mind
python -m examples.spatial_tokens.decode --ckpt models/S-Mind/qwen3.5-4b/rl --gallery --out runs/gallery
python -m examples.spatial_tokens.decode --ckpt models/S-Mind/qwen3.5-4b/rl --set sparbench --question_id sparbench_5491 \
    --out runs/toy_room
python -m examples.spatial_tokens.decode --ckpt models/S-Mind/qwen3.5-4b/rl --out runs/mine \
    --images view1.jpg view2.jpg --question "..." --options "A. ..." "B. ..."
```

Open `viewer.html` in a browser: drag to look around, switch between z<sub>0</sub> and each z<sub>k</sub> (the points move
from one to the next), or play them in order.
