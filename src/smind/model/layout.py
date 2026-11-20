from dataclasses import dataclass

Z_TRIGGER = "<|z|>"
Z_SLOT = "<|zslot|>"
RESERVED_TOKENS = [Z_TRIGGER, Z_SLOT]


@dataclass(frozen=True)
class Layout:
    per_image: int
    camera: int

    def __post_init__(self):
        if self.per_image < 1 or self.camera < 1:
            raise ValueError(f"a layout needs at least one token per image and one camera token, not {self}")

    @classmethod
    def parse(cls, text):
        m, c = map(int, text.split(","))
        return cls(m, c)

    def __str__(self):
        return f"{self.per_image},{self.camera}"

    @property
    def scene_tokens(self):
        return [f"<|scene_{i}|>" for i in range(self.per_image)]

    @property
    def camera_tokens(self):
        return [f"<|cam_{i}|>" for i in range(self.camera)]

    @property
    def tokens(self):
        return self.scene_tokens + self.camera_tokens

    def run(self, n_images):
        return "".join(self.scene_tokens) * n_images + "".join(self.camera_tokens)

    def run_length(self, n_images):
        return n_images * self.per_image + self.camera

    def insert(self, user_text):
        cut = user_text.rfind("<image>") + len("<image>")
        if cut < len("<image>"):
            raise ValueError("no image placeholder")
        return user_text[:cut] + self.run(user_text.count("<image>")) + user_text[cut:]

    def ids(self, tokenizer):
        ids = tokenizer.convert_tokens_to_ids(self.tokens)
        if None in ids or tokenizer.unk_token_id in ids:
            raise ValueError(f"the tokenizer lacks some of the {self} layout's tokens {self.tokens}")
        return ids


def add_tokens(tokenizer, tokens, vocab_rows):
    tokenizer.add_tokens(tokens, special_tokens=True)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if max(ids) >= vocab_rows:
        raise ValueError(f"token id {max(ids)} does not fit the {vocab_rows} embedding rows")
    return ids


def register(tokenizer, layout, vocab_rows):
    ids = add_tokens(tokenizer, layout.tokens, vocab_rows)
    return ids, ids + add_tokens(tokenizer, RESERVED_TOKENS, vocab_rows)
