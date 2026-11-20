import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

from smind.data.build import OUT, RAW, materialize, write

SPAR = RAW / "SPAR-7M/scannet/qa_jsonl/train"
SCANNET = RAW / "ScanNet"
TASKS = {"obj_spatial_relation_oc_mv": 700, "obj_spatial_relation_oo_mv": 700, "spatial_imagination_oc_mv": 700,
         "spatial_imagination_oo_mv": 500, "camera_motion_infer": 500, "position_matching": 400,
         "distance_infer_center_oo_mv": 500}
DEV_PER_TASK = 50
PER_SCENE_TASK = 5
N_TRAIN_SCENES, N_DEV_SCENES = 180, 20
VAL_SCENES = {"scene0217_00", "scene0685_00", "scene0685_01", "scene0685_02"}
INSTR = re.compile(r"\s*Your answer can only include one of (the )?options? [A-H][^\n]*$")
OPT_LINE = re.compile(r"^([A-H])\.\s*(.*)$")

room = lambda s: s.rsplit("_", 1)[0]


def scene_of(rec):
    return rec["image"][0].split("/")[0]


def parse(rec):
    text = INSTR.sub("", rec["conversations"][0]["value"]).rstrip()
    lines = text.split("\n")
    opts = []
    while lines and OPT_LINE.match(lines[-1].strip()):
        opts.insert(0, lines.pop().strip())
    letters = [OPT_LINE.match(o).group(1) for o in opts]
    assert letters == [chr(65 + i) for i in range(len(opts))] and len(opts) >= 2, (rec["id"], text[-300:])
    return "\n".join(lines).strip(), [f"{l}. {OPT_LINE.match(o).group(2).strip()}" for l, o in zip(letters, opts)]


def load_pool():
    pool = {}
    for t in TASKS:
        (f,) = list((SPAR / t / "select").glob("*.jsonl"))
        pool[t] = [json.loads(l) for l in open(f)]
    return pool


def select(seed):
    rng = random.Random(seed)
    pool = load_pool()
    sl = [json.loads(l) for p in ("spatialladder_train", "spatialladder_dev") for l in open(OUT / f"{p}.jsonl")]
    sl_train_rooms = {room(r["scene_id"]) for r in sl if r["split"] == "train"}
    sl_dev_rooms = {room(r["scene_id"]) for r in sl if r["split"] == "dev"}
    by = {t: collections.defaultdict(list) for t in TASKS}
    for t, rs in pool.items():
        for r in rs:
            s = scene_of(r)
            if s not in VAL_SCENES and r["conversations"][1]["value"].strip() in "ABCDEFGH":
                by[t][s].append(r)
    scenes = sorted(set.intersection(*[set(v) for v in by.values()]))
    rooms = sorted({room(s) for s in scenes})
    rng.shuffle(rooms)
    dev_scenes, train_scenes = [], []
    for rm in rooms:
        ss = sorted(s for s in scenes if room(s) == rm)
        if len(dev_scenes) < N_DEV_SCENES and rm not in sl_train_rooms:
            dev_scenes += ss
        elif len(train_scenes) < N_TRAIN_SCENES and rm not in sl_dev_rooms:
            train_scenes += ss
        if len(dev_scenes) >= N_DEV_SCENES and len(train_scenes) >= N_TRAIN_SCENES:
            break

    def spread(t, ss, k):
        lists = {s: rng.sample(by[t][s], len(by[t][s]))[:PER_SCENE_TASK] for s in ss}
        order = list(ss)
        rng.shuffle(order)
        out, rnd = [], 0
        while len(out) < k:
            took = False
            for s in order:
                if rnd < len(lists[s]) and len(out) < k:
                    out.append(lists[s][rnd])
                    took = True
            if not took:
                raise ValueError(f"{t}: only {len(out)} available, need {k}")
            rnd += 1
        return out

    sel = []
    for t, k in TASKS.items():
        sel += [dict(r, split="train", task=t) for r in spread(t, train_scenes, k)]
        sel += [dict(r, split="dev", task=t) for r in spread(t, dev_scenes, DEV_PER_TASK)]
    frames = collections.defaultdict(set)
    for r in sel:
        for p in r["image"]:
            s, _, f = p.split("/")
            frames[s].add(int(f.split(".")[0]))
    write("_spar_selected", sel)
    SCANNET.mkdir(parents=True, exist_ok=True)
    (SCANNET / "spar_frames.json").write_text(json.dumps({s: sorted(v) for s, v in sorted(frames.items())}))
    stats = dict(train_scenes=len(train_scenes), dev_scenes=len(dev_scenes), scenes_with_frames=len(frames),
                 frames=sum(len(v) for v in frames.values()),
                 per_split_task={sp: collections.Counter(r["task"] for r in sel if r["split"] == sp)
                                 for sp in ("train", "dev")},
                 answers={sp: collections.Counter(r["conversations"][1]["value"] for r in sel if r["split"] == sp)
                          for sp in ("train", "dev")})
    print(json.dumps(stats, indent=1))


def build(workers):
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third_party/spar/datasets"))
    from draw_marker import DRAW_FUNCTIONS

    marked_dir = SCANNET / "spar_marked"

    def one(r):
        images = [Image.open(SCANNET / "frames" / p.replace("image_color", "color")).convert("RGB") for p in r["image"]]
        DRAW_FUNCTIONS[r["type"]](images, r)
        raw = []
        for i, im in enumerate(images):
            dst = marked_dir / r["id"].split("_")[0] / f"{r['id']}_{r['task']}_{i}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                im.save(dst)
            raw.append(str(dst))
        stem, opts = parse(r)
        markers = {k: r[k] for k in r if k.endswith(("_bbox", "_point", "_img_idx"))}
        return dict(source="spar", task=r["task"], scene_id=scene_of(r), question_id=f"spar_{r['task']}_{r['id']}",
                    raw_images=raw, question=stem, options=opts, answer=r["conversations"][1]["value"].strip(),
                    split=r["split"], spar_image=r["image"], markers=markers,
                    images=[materialize(Path(p), len(raw)) for p in raw])

    sel = [json.loads(l) for l in open(OUT / "_spar_selected.jsonl")]
    with ThreadPoolExecutor(workers) as ex:
        recs = list(ex.map(one, sel))
    assert len({r["question_id"] for r in recs}) == len(recs)
    for sp in ("train", "dev"):
        write(f"spar_{sp}", [r for r in recs if r["split"] == sp])
    print(json.dumps({sp: collections.Counter(r["task"] for r in recs if r["split"] == sp) for sp in ("train", "dev")}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "build"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    select(args.seed) if args.cmd == "select" else build(args.workers)


if __name__ == "__main__":
    main()
