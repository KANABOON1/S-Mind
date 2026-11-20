import argparse
import base64
import io
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForImageTextToText, AutoProcessor, LogitsProcessorList

from smind.eval.sets import SETS, set_path
from smind.model.checkpoint import SpatialCheckpoint
from smind.model.insertion import question_text
from smind.model.reader import states_of
from smind.model.views import camera_poses

ANSWER = re.compile(r"<answer>\s*([A-Z])\s*</answer>")
CAM = [(242, 183, 91), (98, 198, 227), (238, 134, 168), (143, 209, 138), (199, 162, 245), (240, 143, 107),
       (127, 179, 240), (227, 211, 107)]
VIOLET, INK, INK3 = (183, 163, 245), (236, 233, 246), (138, 134, 160)
PANEL, SS = 520, 2
VIEWER = Path(__file__).with_name("viewer.html")
GALLERY = Path(__file__).with_name("gallery.jsonl")


class Model:

    def __init__(self, path):
        self.ckpt = SpatialCheckpoint.load(path)
        if self.ckpt.insertion is None or not self.ckpt.has_readout:
            raise ValueError(f"{path}: a perspective-stage or RL checkpoint with its readout is required")
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.tok = self.processor.tokenizer
        self.lm = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                              local_files_only=True).cuda().eval()
        self.head = self.ckpt.readout()[1].cuda().eval()
        self.layers = self.lm.model.language_model.layers
        self.scene = torch.tensor(self.ckpt.token_ids(self.tok))
        self.stop = [self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")]

    def inputs(self, rec, images):
        content = [dict(type="image") for _ in images] + [dict(type="text", text=question_text(rec, self.ckpt.layout))]
        prompt = self.tok.apply_chat_template([dict(role="user", content=content)], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
        enc = self.processor(text=[prompt], images=images, return_tensors="pt")
        return enc["input_ids"][0], {k: enc[k].cuda() for k in ("pixel_values", "image_grid_thw")}

    def kwargs(self, ids, media):
        return dict(input_ids=ids[None].cuda(), attention_mask=torch.ones_like(ids)[None].cuda(),
                    mm_token_type_ids=(ids == self.lm.config.image_token_id).long()[None].cuda(), **media)

    def generate(self, ids, media, run, max_new_tokens):
        in_run = set(run.tolist())

        def complete_run(input_ids, scores):
            d = 0
            while d < len(run) and int(input_ids[0, -1 - d]) in in_run:
                d += 1
            if 0 < d < len(run):
                forced = torch.full_like(scores, float("-inf"))
                forced[0, run[d]] = 0
                return forced
            return scores

        with torch.no_grad():
            return self.lm.generate(**self.kwargs(ids, media), do_sample=False, num_beams=1,
                                    max_new_tokens=max_new_tokens, eos_token_id=self.stop,
                                    logits_processor=LogitsProcessorList([complete_run]))[0].cpu()

    def states(self, ids, media):
        kept = {}
        grab = [layer.register_forward_hook(lambda m, a, o, j=j: kept.__setitem__(j, states_of(o)[0].detach()))
                for j, layer in enumerate(self.layers)]
        with torch.no_grad():
            self.lm.model(**self.kwargs(ids, media), use_cache=False)
        for g in grab:
            g.remove()
        return torch.stack([kept[j] for j in range(len(self.layers))])

    def read(self, states, grid):
        view = dict(n_views=len(grid), grid=grid, view_index=torch.arange(1, len(grid), device="cuda"))
        with torch.no_grad():
            out = self.head.read(self.head.belief(states.cuda()).float(), view)
        return {k: v.float().cpu() for k, v in out.items()}

    def text(self, ids):
        return self.tok.decode([t for t in ids.tolist() if t not in set(self.scene.tolist())], skip_special_tokens=True)


def patch_colors(images, grid):
    return np.concatenate([np.asarray(im.resize((c, r), Image.BOX), np.uint8).reshape(-1, 3)
                           for im, (r, c) in zip(images, grid)])


def write_ply(path, points, colors):
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")


def font(size):
    return ImageFont.load_default(size=size * SS)


def stage(w, h):
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - w / 2) / (0.6 * w), (yy - 0.38 * h) / (0.45 * h)).clip(0, 1)[..., None]
    inner, mid, outer = np.array([27, 30, 44]), np.array([13, 14, 21]), np.array([7, 8, 12])
    rgb = np.where(r < 0.55, inner + (mid - inner) * (r / 0.55), mid + (outer - mid) * ((r - 0.55) / 0.45))
    return Image.fromarray(rgb.astype(np.uint8))


def wedge(d, center, phi, size, color):
    x, y = center
    f, s = np.array([math.sin(phi), -math.cos(phi)]), np.array([math.cos(phi), math.sin(phi)])
    tip, back = np.array([x, y]) + f * size, np.array([x, y]) - f * size * 0.35
    d.polygon([tuple(tip), tuple(back + s * size * 0.55), tuple(back - s * size * 0.55)], fill=color)


def panel(points, colors, title, label, cameras=None):
    W, H = PANEL * SS, (PANEL + 56) * SS
    img = stage(W, H)
    d = ImageDraw.Draw(img)
    xy = points[:, :2]
    extent = max(float(np.percentile(np.hypot(*xy.T), 92)) * 1.15, 1e-3)
    if cameras is not None:
        extent = max(extent, float(np.hypot(*cameras[:, :2].T).max()) * 1.2)
    top = 56 * SS
    px = lambda x, y: (W / 2 + x / extent * W * 0.46, top + (H - top) / 2 - y / extent * (H - top) * 0.46)
    d.ellipse([px(-extent, extent)[0], px(-extent, extent)[1], px(extent, -extent)[0], px(extent, -extent)[1]],
              outline=(43, 45, 61), width=SS)
    order = np.argsort(points[:, 2])
    r = 2.6 * SS
    for i in order:
        u, v = px(*xy[i])
        d.ellipse([u - r, v - r, u + r, v + r], fill=tuple(int(c) for c in colors[i]))
    if cameras is not None:
        for k, (x, y, phi) in enumerate(cameras):
            wedge(d, px(x, y), phi, 15 * SS, CAM[k % len(CAM)])
            u, v = px(x, y)
            d.text((u + 12 * SS, v - 22 * SS), str(k + 1), fill=CAM[k % len(CAM)], font=font(15))
    else:
        wedge(d, px(0, 0), 0.0, 17 * SS, VIOLET)
    d.text((18 * SS, 14 * SS), label, fill=VIOLET, font=font(15))
    d.text((18 * SS + d.textlength(label + "  ", font=font(15)), 14 * SS), title[:58], fill=INK, font=font(15))
    d.text((18 * SS, 34 * SS), "seen from above", fill=INK3, font=font(12))
    return img.resize((PANEL, PANEL + 56), Image.LANCZOS)


def photo_uri(im, side=640):
    im = im.copy()
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def decode(model, rec, out, args):
    out.mkdir(parents=True, exist_ok=True)
    images = [Image.open(p).convert("RGB") for p in rec["images"]]
    ids, media = model.inputs(rec, images)
    grid = [(int(h) // 2, int(w) // 2) for _, h, w in media["image_grid_thw"].tolist()]
    R = model.ckpt.layout.run_length(len(images))
    run = ids[torch.isin(ids, model.scene)][:R]
    full = model.generate(ids, media, run, args.max_new_tokens)
    pos = torch.isin(full, model.scene).nonzero()[:, 0].tolist()
    starts = [pos[k] for k in range(0, len(pos) - len(pos) % R, R)]
    states = model.states(full, media)
    response = model.text(full[len(ids):])
    lines = [l.strip() for l in response.split("\n") if l.strip() and not ANSWER.fullmatch(l.strip())]
    colors = patch_colors(images, grid)
    result = dict(question=rec["question"], options=rec["options"], response=response,
                  answer=(m.group(1) if (m := ANSWER.search(response)) else None), states=[])
    page, panels, keep = [], [], None
    for k, s in enumerate(starts):
        read = model.read(states[:, s:s + R], grid)
        if keep is None:
            conf = read.get("conf", torch.ones(len(read["points"])))
            keep = (conf >= conf.quantile(0.2)).numpy()
        points = read["points"].numpy()
        if k == 0:
            name, label, title, sentence = "z0", "z0", "the scene after the images", -1
            cameras = camera_poses(read["camera"]).numpy()
        else:
            before = model.text(full[len(ids):s])
            title = before.strip().split("\n")[-1]
            name, label, cameras = f"z{k}", f"z{k}", None
            sentence = len([l for l in before.split("\n")[:-1] if l.strip()])
        write_ply(out / f"{name}.ply", points[keep], colors[keep])
        panels.append(panel(points[keep], colors[keep], title, label, cameras))
        page.append(dict(points=points.round(4).tolist(), keep=keep.tolist(), viewpoint=title, sentence=sentence,
                         cameras=None if cameras is None else cameras.round(4).tolist()))
        result["states"].append(dict(name=name, viewpoint=None if k == 0 else title,
                                     cameras=None if cameras is None else cameras.round(3).tolist()))
    sheet = Image.new("RGB", (PANEL * len(panels) + 6 * (len(panels) - 1), PANEL + 56), (18, 17, 26))
    for k, p in enumerate(panels):
        sheet.paste(p, (k * (PANEL + 6), 0))
    sheet.save(out / "topdown.png")
    data = dict(title=rec.get("title", ""), question=rec["question"], options=rec["options"], answer=result["answer"], sentences=lines, grid=grid,
                photos=[photo_uri(im) for im in images], states=page)
    (out / "viewer.html").write_text(VIEWER.read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))))
    (out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(response)
    print(f"answer {result['answer']}")
    print(f"wrote {out}/viewer.html, {out}/topdown.png, {out}/z*.ply, {out}/result.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="an S-Mind checkpoint after the perspective stage or RL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", nargs="+")
    ap.add_argument("--question")
    ap.add_argument("--options", nargs="+", help='e.g. "A. Left" "B. Right"')
    ap.add_argument("--set", choices=sorted(SETS), help="a built benchmark or split, with --question_id")
    ap.add_argument("--question_id")
    ap.add_argument("--gallery", action="store_true", help="every question of gallery.jsonl, one folder each")
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    args = ap.parse_args()
    if args.gallery:
        picks = [json.loads(line) for line in open(GALLERY)]
    elif args.set and args.question_id:
        picks = [dict(set=args.set, question_id=args.question_id)]
    elif args.images and args.question and args.options:
        picks = [dict(images=args.images, question=args.question, options=args.options)]
    else:
        ap.error("give --images, --question and --options; or --set and --question_id; or --gallery")
    model = Model(args.ckpt)
    for pick in picks:
        rec = pick
        if "set" in pick:
            rec = next((r for r in map(json.loads, open(set_path(pick["set"]))) if r["question_id"] == pick["question_id"]),
                       None)
            if rec is None:
                raise SystemExit(f"{pick['question_id']} is not in {set_path(pick['set'])}")
            rec = dict(rec, title=pick.get("title", ""))
        out = Path(args.out) / pick["question_id"] if len(picks) > 1 else Path(args.out)
        decode(model, rec, out, args)


if __name__ == "__main__":
    main()
