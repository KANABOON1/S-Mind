import json
import os
from dataclasses import dataclass
from pathlib import Path

from safetensors.torch import load_file, save_file

from smind.model.layout import RESERVED_TOKENS, Layout
from smind.model.readout import READOUTS

FORMAT = "smind-ckpt/1"
SPATIAL_CONFIG = "spatial_config.json"
READOUT = "readout"
INSERTIONS = ("self",)


def readout_config(name, layers, hidden, args, scene_ids):
    return {"class": name, "layers": layers, "hidden": hidden, "args": args, "scene_ids": list(scene_ids)}


def build_readout(config):
    return READOUTS[config["class"]](config["layers"], config["hidden"], **config["args"])


def save_readout(path, config, head):
    path = Path(path) / READOUT
    path.mkdir(parents=True, exist_ok=False)
    (path / "config.json").write_text(json.dumps(config, indent=1))
    save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, path / "model.safetensors")


def write_spatial(path, layout, insertion):
    if insertion not in INSERTIONS + (None,):
        raise ValueError(f"insertion {insertion!r}")
    config = {"format": FORMAT, "layout": {"per_image": layout.per_image, "camera": layout.camera},
              "scene_tokens": layout.scene_tokens, "camera_tokens": layout.camera_tokens,
              "reserved_tokens": RESERVED_TOKENS, "insertion": insertion}
    (Path(path) / SPATIAL_CONFIG).write_text(json.dumps(config, indent=1))


def link_model(src, dst):
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=False)
    for f in sorted(Path(src).iterdir()):
        if f.is_file() and f.name != SPATIAL_CONFIG:
            os.link(f, dst / f.name)


def save(path, model, processor, layout, insertion, readout=None):
    model.save_pretrained(path)
    processor.save_pretrained(path)
    write_spatial(path, layout, insertion)
    if readout is not None:
        save_readout(path, *readout)


@dataclass(frozen=True)
class SpatialCheckpoint:
    path: Path
    layout: Layout
    insertion: str | None

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not (path / SPATIAL_CONFIG).exists():
            raise FileNotFoundError(f"{path} has no {SPATIAL_CONFIG}: not an S-Mind checkpoint")
        config = json.loads((path / SPATIAL_CONFIG).read_text())
        if config["format"] != FORMAT:
            raise ValueError(f"{path}: format {config['format']}, this code reads {FORMAT}")
        layout = Layout(**config["layout"])
        if (config["scene_tokens"], config["camera_tokens"], config["reserved_tokens"]) != (
                layout.scene_tokens, layout.camera_tokens, RESERVED_TOKENS):
            raise ValueError(f"{path}: token strings that are not the {layout} layout's")
        if config["insertion"] not in INSERTIONS + (None,):
            raise ValueError(f"{path}: insertion {config['insertion']!r}")
        return cls(path, layout, config["insertion"])

    @property
    def has_readout(self):
        return (self.path / READOUT).is_dir()

    def readout_config(self):
        return json.loads((self.path / READOUT / "config.json").read_text())

    def readout(self):
        config = self.readout_config()
        head = build_readout(config)
        head.load_state_dict(load_file(self.path / READOUT / "model.safetensors"), strict=True)
        return config, head

    def token_ids(self, tokenizer):
        ids = self.layout.ids(tokenizer)
        if self.has_readout and self.readout_config()["scene_ids"] != ids:
            raise ValueError(f"{self.path}: the readout was trained with ids {self.readout_config()['scene_ids']}, "
                             f"the tokenizer has {ids}")
        return ids
