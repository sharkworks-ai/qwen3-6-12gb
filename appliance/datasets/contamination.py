from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Iterable


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def shingles(text: str, width: int = 8) -> set[str]:
    words = normalize(text).split()
    return {
        hashlib.sha1(" ".join(words[i : i + width]).encode()).hexdigest()
        for i in range(max(0, len(words) - width + 1))
    }


def overlap_score(a: str, b: str) -> float:
    sa, sb = shingles(a), shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def flag_overlaps(
    training: Iterable[tuple[str, str]],
    heldout: Iterable[tuple[str, str]],
    threshold: float = 0.25,
) -> list[dict]:
    flagged = []
    held = list(heldout)
    for train_id, train_text in training:
        for eval_id, eval_text in held:
            score = overlap_score(train_text, eval_text)
            if score >= threshold:
                flagged.append(
                    {
                        "training_id": train_id,
                        "evaluation_id": eval_id,
                        "overlap": score,
                    }
                )
    return flagged
