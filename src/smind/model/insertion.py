import torch

from smind import formats


def question_text(rec, layout):
    return layout.run(len(rec["images"])) + formats.user_text(rec, image_token="")


def inserted_mask(ids, prev, token_ids):
    return torch.isin(ids, token_ids) & ~((ids == token_ids[0]) & ~torch.isin(prev, token_ids))


class Insertion:

    def __init__(self, layout, tokenizer):
        self.layout, self.tok = layout, tokenizer
        self.token_ids = layout.ids(tokenizer)
        self.inserted = set(self.token_ids)
        self.stop_ids = {self.token_ids[0]}

    def run_ids(self, n_images):
        return self.tok.convert_tokens_to_ids(self.tok.tokenize(self.layout.run(n_images)))

    def text_ids(self, answer):
        return [t for t in answer if t not in self.inserted]

    def due(self, answer, n_images):
        return self.run_ids(n_images)[1:]
