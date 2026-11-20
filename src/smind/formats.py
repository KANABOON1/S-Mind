import re

SUFFIX = (
    "Please reason step by step first, then give your final answer as a single option letter "
    "in the format <answer>X</answer>."
)

_ANSWER_RE = re.compile(r"<answer>\s*\(?([A-Z])\)?(?:\s*[.:)]|\s*</answer>|\s)", re.S)


def user_text(rec, image_token="<image>"):
    images = image_token * len(rec["images"])
    options = "\n".join(rec["options"])
    return f"{images}{rec['question']}\n{options}\n{SUFFIX}"


def extract_answer(text, n_options=None):
    if text is None:
        return None
    hits = _ANSWER_RE.findall(text + " ")
    if not hits:
        return None
    letter = hits[-1]
    if n_options is not None and ord(letter) - ord("A") >= n_options:
        return None
    return letter
