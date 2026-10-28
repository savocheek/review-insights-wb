"""Review → short phrases ("одна мысль на фразу"), the same splitting as sentiment.py of the reviews project.

- clauses():  sentences, contrast clauses ("…, но …") and comma-separated parts, at least 2 words each
- norm():     key for deduplication (lower case, ё → е, no punctuation)
- polarity(): sign of a phrase from sentiment probabilities, ignoring the neutral mass (short phrases lean
              "neutral" because the model learnt 3★ = neutral from whole reviews)
"""
import html
import re

_CLAUSE_SPLIT = re.compile(r"(?<=[.!?…;])\s+|\s*[,–—-]\s*(?=(?:но|а|однако|зато|хотя|правда)\s)|\s+(?=но\s)", re.I)
_NON_WORD = re.compile(r"[^\w\s]+")


def clean(text):
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text).replace("\\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def clauses(text, max_words=30):
    parts = [c for p in _CLAUSE_SPLIT.split(clean(text)) if p for c in p.split(",")]
    parts = [p.strip(" ,.;!?…()\"'«»") for p in parts]
    return [p for p in parts if 2 <= len(p.split()) <= max_words]


def norm(phrase):
    return " ".join(_NON_WORD.sub(" ", phrase.lower().replace("ё", "е")).split())


def polarity(probs):
    """probs: (n, 3) negative/neutral/positive → (P(pos) − P(neg)) / (P(pos) + P(neg)) ∈ [−1, 1]."""
    return (probs[:, 2] - probs[:, 0]) / (probs[:, 2] + probs[:, 0] + 1e-6)
