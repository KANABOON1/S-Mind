import math

import torch


def viewer_frame(points, pose):
    c, s = torch.cos(pose[2]), torch.sin(pose[2])
    d = points[:, :2] - pose[:2]
    return torch.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c, points[:, 2]], -1)


def camera_poses(camera):
    phi = torch.atan2(camera[:, 2], camera[:, 3])
    return torch.cat([torch.zeros(1, 3, device=camera.device), torch.cat([camera[:, :2], phi[:, None]], -1)])


def resolve(specs, cams, objects):
    def facing(spec, pos):
        j, m = spec.get("facing_object"), spec.get("facing_image")
        if j is not None:
            if objects is None:
                return None
            d = objects[j] - pos[:2]
            return torch.atan2(d[0], d[1])[None]
        if m is not None:
            return cams[m][2:] + math.radians(spec.get("facing_turn_deg", 0.0))
        return None

    out, cur = [], cams[0]
    for spec in specs:
        kind = spec["kind"]
        if kind == "camera":
            cur = cams[spec["image"]]
        elif kind == "camera_facing":
            phi = facing(spec, cams[spec["image"]])
            cur = None if phi is None else torch.cat([cams[spec["image"]][:2], phi])
        elif kind == "object":
            pos = None if objects is None else objects[spec["at_object"]]
            phi = None if pos is None else facing(spec, pos)
            cur = None if phi is None else torch.cat([pos, phi])
        elif kind == "relative":
            base = cams[spec["from_image"]] if spec.get("from_image") is not None else cur
            if base is not None:
                phi = facing(spec, base)
                cur = torch.cat([base[:2], base[2:] + math.radians(spec["turn_deg"]) if phi is None else phi])
        elif kind == "unresolved":
            cur = None
        elif kind != "keep":
            raise ValueError(f"view action {kind}")
        out.append(cur)
    return out
