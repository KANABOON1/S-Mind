import json
import math
from pathlib import Path

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


POINT_NEAR, POINT_FAR = 0.1, 0.25
BANDS = 6


TOKEN_DIR = Path("data/smind/geometry/token")


def attach_token_points(maps, files=None):
    if files is None:
        files = sorted(TOKEN_DIR.glob("points_*.npz"))
        if not files:
            raise FileNotFoundError(f"no points_*.npz in {TOKEN_DIR}: build the geometry supervision first (README, Data)")
    table = {}
    for f in files:
        z = np.load(f)
        table.update(zip(z["set_ids"].tolist(), zip(z["grid"], z["points"])))
    for m in maps:
        if m["set_id"] in table:
            grid, pts = table[m["set_id"]]
            grid = [(int(r), int(c)) for r, c in grid[:len(m["views"])]]
            m["grid"] = grid
            m["points"] = np.concatenate([pts[k, :r * c] for k, (r, c) in enumerate(grid)]).astype(np.float32)
    return maps


class GroupedGeometry(nn.Module):

    def __init__(self, layers, hidden, dc=512, d=256, m=None, c=None, conf_alpha=None, local_points=False,
                 local_conf=False):
        super().__init__()
        if not m or not c:
            raise ValueError("geometry_group needs the group sizes m and c")
        if local_conf and (conf_alpha is None or not local_points):
            raise ValueError("local_conf needs conf_alpha and local_points")
        self.m, self.c = m, c
        self.conf_alpha, self.local_points, self.local_conf = conf_alpha, local_points, local_conf
        self.layer_logits = nn.Parameter(torch.zeros(layers))
        self.proj = nn.Linear(hidden, dc)
        self.query_norm, self.memory_norm, self.out_norm = nn.LayerNorm(d), nn.LayerNorm(dc), nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, 8, kdim=dc, vdim=dc, add_bias_kv=True, batch_first=True)
        self.patch_query = nn.Linear(4 * BANDS, d)
        self.point = nn.Linear(d, 3 + (conf_alpha is not None) + 3 * local_points + local_conf)
        self.camera = nn.Linear(d, 4)
        self.camera_query = nn.Sequential(nn.LayerNorm(dc), nn.Linear(dc, d), nn.GELU(), nn.Linear(d, d))

    def belief(self, states):
        w = torch.softmax(self.layer_logits, 0)
        return self.proj((w[:, None, None] * states.float()).sum(0))

    def code(self, rows, cols, device):
        half = max(rows, cols) / 2
        v, u = torch.meshgrid((torch.arange(rows, device=device) + 0.5 - rows / 2) / half,
                              (torch.arange(cols, device=device) + 0.5 - cols / 2) / half, indexing="ij")
        a = torch.stack([u.flatten(), v.flatten()], -1)[..., None] * (2.0 ** torch.arange(BANDS, device=device)) * math.pi
        return torch.cat([a.sin(), a.cos()], -1).flatten(-2)

    def attend(self, z, q):
        m = self.memory_norm(z)[None]
        return self.cross(q[None], m, m, need_weights=False)[0][0]

    def read(self, z, t):
        n = t["n_views"]
        if z.shape[0] != n * self.m + self.c:
            raise ValueError(f"{z.shape[0]} scene states for {n} images in groups of {self.m} and {self.c} cameras")
        cam = z[n * self.m:]
        groups = z[:n * self.m].reshape(n, self.m, -1)
        h = [self.attend(torch.cat([groups[k], cam]), self.query_norm(self.patch_query(self.code(r, c, z.device))))
             for k, (r, c) in enumerate(t["grid"])]
        views = t["view_index"]
        q = self.camera_query(groups[views].mean(1))
        h.append(self.attend(z, self.query_norm(q)))
        h = self.out_norm(torch.cat(h))
        cells = sum(r * c for r, c in t["grid"])
        return self.split(self.point(h[:cells]), self.camera(h[cells:]))

    def split(self, out, camera):
        pred = dict(points=out[:, :3], camera=camera)
        if self.conf_alpha is not None:
            pred["conf"] = 1 + torch.exp(out[:, 3].clamp(max=20))
        if self.local_points:
            at = 3 + (self.conf_alpha is not None)
            pred["local"] = out[:, at:at + 3]
        if self.local_conf:
            pred["conf_local"] = 1 + torch.exp(out[:, at + 3].clamp(max=20))
        return pred

    def local_targets(self, t):
        views = self.cell_views(t)[t["seen"]]
        cells = t["seen"].nonzero()[:, 0]
        want = t["points"].clone()
        later = views > 0
        if "xy" not in t:
            return cells[~later], want[~later]
        f = F.normalize(t["facing"], dim=-1)[views[later] - 1]
        d = want[later, :2] - t["xy"][views[later] - 1]
        want[later, 0] = d[:, 0] * f[:, 1] - d[:, 1] * f[:, 0]
        want[later, 1] = (d * f).sum(-1)
        return cells, want

    def targets(self, m, device):
        views = [k for k in range(1, len(m["views"]))]
        t = dict(n_views=len(m["views"]), view_index=torch.tensor(views, dtype=torch.long, device=device),
                 facing=torch.tensor([m["views"][k]["facing"] for k in views], dtype=torch.float32,
                                     device=device).reshape(-1, 2))
        raw = torch.tensor([m["views"][k]["xy"] for k in views], dtype=torch.float32, device=device).reshape(-1, 2)
        if "points" in m:
            p = torch.as_tensor(m["points"], device=device).reshape(-1, 3)
            seen = torch.isfinite(p[:, 0])
            scale = p[seen].norm(dim=-1).mean()
            t.update(seen=seen, points=p[seen] / scale, scale=scale, xy=raw / scale, raw_xy=raw)
        t["grid"] = m.get("grid", [])
        return t

    def loss(self, pred, t):
        loss = 0.0
        if "points" in t and len(t["points"]):
            got, want = pred["points"][t["seen"]], t["points"]
            err = F.smooth_l1_loss(got, want, beta=0.1, reduction="none").sum(-1)
            if self.conf_alpha is not None:
                c = pred["conf"][t["seen"]]
                err = c * err - self.conf_alpha * torch.log(c)
            loss = err.mean()
            if self.local_points:
                cells, want = self.local_targets(t)
                if len(cells):
                    err = F.smooth_l1_loss(pred["local"][cells], want, beta=0.1, reduction="none").sum(-1)
                    if self.local_conf:
                        c = pred["conf_local"][cells]
                        err = c * err - self.conf_alpha * torch.log(c)
                    loss = loss + err.mean()
        if len(t["view_index"]):
            target = torch.cat([t["xy"], t["facing"]], -1) if "xy" in t else t["facing"]
            got = pred["camera"] if "xy" in t else pred["camera"][:, 2:]
            loss = loss + F.smooth_l1_loss(got, target, beta=0.1, reduction="none").sum(-1).mean()
        return loss

    def cross_pairs(self, t):
        dev = t["view_index"].device
        (r1, c1), offsets = t["grid"][0], np.cumsum([0] + [r * c for r, c in t["grid"]])
        first, other = [], []
        for k, (r, c) in enumerate(t["grid"][1:], 1):
            rows, cols = torch.meshgrid(torch.arange(r, device=dev), torch.arange(c, device=dev), indexing="ij")
            to1 = ((rows.flatten() + 0.5) / r * r1).long() * c1 + ((cols.flatten() + 0.5) / c * c1).long()
            first.append(offsets[0] + to1)
            other.append(offsets[k] + torch.arange(r * c, device=dev))
        empty = torch.zeros(0, dtype=torch.long, device=dev)
        return (torch.cat(first) if first else empty), (torch.cat(other) if other else empty)

    def cell_views(self, t):
        sizes = torch.tensor([r * c for r, c in t["grid"]], device=t["view_index"].device)
        return torch.arange(len(sizes), device=sizes.device).repeat_interleave(sizes)

    @torch.no_grad()
    def metrics(self, pred, t):
        out = {}
        if "points" in t and len(t["points"]):
            err = (pred["points"][t["seen"]] - t["points"]).norm(dim=-1)
            out["point_near"] = int((err <= POINT_NEAR).sum()), len(err)
            out["point_far"] = int((err <= POINT_FAR).sum()), len(err)
            dist = t["points"].norm(dim=-1)
            ratio = pred["points"][t["seen"]].norm(dim=-1) / dist.clamp_min(1e-6)
            out["depth_ok"] = int(((ratio - 1).abs() <= 0.1).sum()), len(ratio)
            far = dist > 1.5
            if far.any():
                out["far_depth_ok"] = int(((ratio[far] - 1).abs() <= 0.2).sum()), int(far.sum())
            first = (self.cell_views(t) == 0)[t["seen"]]
            for tag, sel in (("v1", first), ("vk", ~first)):
                if sel.any():
                    out[f"point_far_{tag}"] = int((err[sel] <= POINT_FAR).sum()), int(sel.sum())
                    out[f"depth_ok_{tag}"] = int(((ratio[sel] - 1).abs() <= 0.1).sum()), int(sel.sum())
            if self.local_points:
                cells, want = self.local_targets(t)
                if len(cells):
                    out["local_far"] = int(((pred["local"][cells] - want).norm(dim=-1) <= POINT_FAR).sum()), len(cells)
            full = torch.full_like(pred["points"], float("nan"))
            full[t["seen"]] = t["points"]
            first, other = self.cross_pairs(t)
            both = torch.isfinite(full[first, 0]) & torch.isfinite(full[other, 0])
            if both.any():
                first, other = first[both], other[both]
                bearing(out, (pred["points"][other] - pred["points"][first])[:, :2],
                        (full[other] - full[first])[:, :2], "cross_bearing")
        if len(t["view_index"]):
            f = F.normalize(pred["camera"][:, 2:], dim=-1)
            out["facing"] = int(((f * t["facing"]).sum(-1) >= math.cos(math.pi / 4) - 1e-6).sum()), len(f)
            if "xy" in t:
                out["view_hit"] = int(((pred["camera"][:, :2] - t["xy"]).norm(dim=-1) <= POINT_FAR).sum()), len(f)
                view_side(out, pred["camera"][:, 0] * t["scale"], t["raw_xy"][:, 0])
        return out


READOUTS = {"geometry_group": GroupedGeometry}


def sector(v):
    a = torch.atan2(v[..., 0], v[..., 1])
    return torch.remainder(torch.round(a / (math.pi / 2)), 4).long()


def bearing(out, pv, tv, key):
    a = torch.atan2(tv[..., 0], tv[..., 1])
    clear = (torch.remainder(a + math.pi / 4, math.pi / 2) - math.pi / 4).abs() < math.pi / 8
    clear = clear & (tv.norm(dim=-1) > 1e-6)
    if clear.any():
        out[key] = ((sector(pv) == sector(tv)) & clear).sum().item(), clear.sum().item()


def view_side(out, px, tx):
    for key, floor in (("view_side", 1e-6), ("view_side_far", 0.3)):
        side = tx.abs() >= floor
        if side.any():
            out[key] = ((torch.sign(px) == torch.sign(tx)) & side).sum().item(), side.sum().item()


def load_maps(path):
    return [json.loads(line) for line in open(path)]
