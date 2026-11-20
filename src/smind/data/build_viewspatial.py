import json
from pathlib import Path

from smind.data.build import RAW, attach_images, write


def build_viewspatial():
    root = RAW / "ViewSpatial-Bench"
    data = json.load(open(root / "ViewSpatial-Bench.json"))
    recs = []
    for i, x in enumerate(data):
        paths = [str(root.parent / p) for p in x["image_path"]]
        assert all(Path(p).exists() for p in paths), paths
        opts = [o.strip() for o in x["choices"].split("\n") if o.strip()]
        opts = [f"{o[0]}. {o[2:].strip()}" for o in opts]
        letter = x["answer"].strip()[0]
        assert letter in "ABCDEFGH"[: len(opts)] and opts["ABCDEFGH".index(letter)][3:] == x["answer"].strip()[3:], x
        recs.append(dict(source="viewspatial", task=x["question_type"], scene_id=Path(paths[0]).parent.parent.name,
                         question_id=f"viewspatial_{i}", raw_images=paths, question=x["question"].strip(),
                         options=opts, answer=letter, split="test"))
    return recs


def main():
    vs = attach_images(build_viewspatial())
    write("bench_viewspatial", vs)
    print(len(vs))


if __name__ == "__main__":
    main()
