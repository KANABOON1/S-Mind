import torch
from torch.nn import functional as F


def states_of(output):
    return output[0] if isinstance(output, tuple) else output


def scene_fill(scene_ids, rows):
    ids = torch.tensor(scene_ids)

    def fill(input_ids):
        mask = torch.isin(input_ids, ids.to(input_ids.device))
        which = (input_ids[mask][:, None] == ids.to(input_ids.device)).int().argmax(1)
        return mask, rows[which]
    return fill


class SceneReader:

    def __init__(self, model, scene_ids, pad_id):
        self.model, self.blocks = model, model.model.language_model.layers
        self.device = model.get_input_embeddings().weight.device
        self.scene, self.pad_id = torch.tensor(scene_ids, dtype=torch.long), pad_id
        self.outputs, self.capture, self.fill = {}, False, None
        for j, block in enumerate(self.blocks):
            block.register_forward_hook(self._keep(j))
        model.get_input_embeddings().register_forward_hook(self._fill)

    def _keep(self, layer):
        def hook(module, inputs, output):
            if self.capture:
                self.outputs[layer] = states_of(output)
        return hook

    def _fill(self, module, inputs, output):
        if self.fill is None:
            return output
        mask, values = self.fill
        output = output.clone()
        output[mask] = values.to(output.dtype)
        return output

    def length(self, item, prefix_only):
        if prefix_only:
            return int(torch.isin(item["input_ids"], self.scene).nonzero()[-1]) + 1
        return len(item["input_ids"])

    def forward(self, items, prefix_only=False, lengths=None, fill=None):
        lengths = lengths or [self.length(x, prefix_only) for x in items]
        width = max(lengths)
        input_ids = torch.full((len(items), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(items), width), dtype=torch.long)
        position = torch.zeros((4, len(items), width), dtype=torch.long)
        media = {}
        for b, (item, n) in enumerate(zip(items, lengths)):
            input_ids[b, :n], mask[b, :n] = item["input_ids"][:n], 1
            position[:, b, :n] = item["position_ids"][:, :n]
            position[:, b, n:] = item["position_ids"][:, n - 1:n] + torch.arange(1, width - n + 1)
            for k, v in item.get("multi_modal_inputs", {}).items():
                media.setdefault(k, []).append(v)
        dev = self.device
        input_ids, mask, position = input_ids.to(dev), mask.to(dev), position.to(dev)
        media = {k: torch.cat(v).to(dev) for k, v in media.items()}
        self.outputs, self.capture, self.fill = {}, True, fill(input_ids) if fill else None
        try:
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                hidden = self.model.model(input_ids=input_ids, attention_mask=mask, position_ids=position,
                                          use_cache=False, **media).last_hidden_state
        finally:
            self.capture, self.fill = False, None
        return hidden, input_ids, mask

    def read(self, positions):
        rows = torch.arange(len(positions), device=positions.device)[:, None]
        return torch.stack([self.outputs[j][rows, positions] for j in range(len(self.blocks))], 1)

    def scene_states(self, input_ids):
        rows = [row.nonzero()[:, 0] for row in torch.isin(input_ids, self.scene.to(input_ids.device))]
        width = max(len(r) for r in rows)
        states = self.read(torch.stack([F.pad(r, (0, width - len(r))) for r in rows]))
        return [x[:, :len(r)] for x, r in zip(states, rows)]
