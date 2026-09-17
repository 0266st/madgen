"""Phoneme inventory (pyopenjtalk's), kana -> phonemes, and the phonetic distance table.

Kana conversion is a plain table so that rendering from a UST/USTX needs no G2P engine;
it produces the same symbols pyopenjtalk.g2p() does for kana.
"""

from __future__ import annotations

from functools import cache

VOWELS = ("a", "i", "u", "e", "o")
MORAIC_NASAL = "N"
CLOSURE = "cl"          # sokuon: a short silence, never sounded
PAUSE = "pau"

# consonant: (place, manner, voiced, palatalized)
CONSONANTS: dict[str, tuple[str, str, bool, bool]] = {
    "k": ("velar", "plosive", False, False),
    "g": ("velar", "plosive", True, False),
    "ky": ("velar", "plosive", False, True),
    "gy": ("velar", "plosive", True, True),
    "s": ("alveolar", "fricative", False, False),
    "z": ("alveolar", "fricative", True, False),
    "sh": ("postalveolar", "fricative", False, True),
    "j": ("postalveolar", "affricate", True, True),
    "t": ("alveolar", "plosive", False, False),
    "d": ("alveolar", "plosive", True, False),
    "ty": ("alveolar", "plosive", False, True),
    "dy": ("alveolar", "plosive", True, True),
    "ts": ("alveolar", "affricate", False, False),
    "ch": ("postalveolar", "affricate", False, True),
    "n": ("alveolar", "nasal", True, False),
    "ny": ("alveolar", "nasal", True, True),
    "h": ("glottal", "fricative", False, False),
    "hy": ("glottal", "fricative", False, True),
    "f": ("bilabial", "fricative", False, False),
    "b": ("bilabial", "plosive", True, False),
    "by": ("bilabial", "plosive", True, True),
    "p": ("bilabial", "plosive", False, False),
    "py": ("bilabial", "plosive", False, True),
    "m": ("bilabial", "nasal", True, False),
    "my": ("bilabial", "nasal", True, True),
    "r": ("alveolar", "liquid", True, False),
    "ry": ("alveolar", "liquid", True, True),
    "y": ("palatal", "approximant", True, True),
    "w": ("bilabial", "approximant", True, False),
    "v": ("bilabial", "fricative", True, False),
}

INVENTORY: tuple[str, ...] = VOWELS + (MORAIC_NASAL,) + tuple(CONSONANTS)

# Vowel pairs that are acoustically close (height / backness neighbours).
_VOWEL_NEAR = {
    frozenset("ie"): 0.4, frozenset("uo"): 0.4, frozenset("iu"): 0.5,
    frozenset("ea"): 0.5, frozenset("oa"): 0.5,
}


def is_voiced_sustained(ph: str) -> bool:
    """Units whose pitch matters: vowels and the moraic nasal."""
    return ph in VOWELS or ph == MORAIC_NASAL


def normalize(ph: str) -> str:
    """pyopenjtalk writes devoiced vowels in upper case (A I U E O); treat them as vowels."""
    return ph.lower() if ph in ("A", "I", "U", "E", "O") else ph


@cache
def distance(a: str, b: str) -> float:
    """0 (same) .. 1 (unrelated)."""
    a, b = normalize(a), normalize(b)
    if a == b:
        return 0.0
    if a in VOWELS and b in VOWELS:
        return _VOWEL_NEAR.get(frozenset(a + b), 0.7)
    if MORAIC_NASAL in (a, b):
        other = b if a == MORAIC_NASAL else a
        if other in ("n", "m", "ny", "my"):
            return 0.4
        if other == "u":
            return 0.6
        return 0.9
    if a in CONSONANTS and b in CONSONANTS:
        pa, ma, va, ya = CONSONANTS[a]
        pb, mb, vb, yb = CONSONANTS[b]
        d = 0.25 * (pa != pb) + 0.3 * (ma != mb) + 0.2 * (va != vb) + 0.1 * (ya != yb)
        return min(0.9, max(0.1, d))
    return 1.0


def nearest(ph: str, n: int) -> list[tuple[str, float]]:
    """The n phonemes closest to `ph` (excluding itself), with their distances."""
    others = sorted((distance(ph, q), q) for q in INVENTORY if q != normalize(ph))
    return [(q, d) for d, q in others[:n]]


def fill_candidates(best: str, confidence: float, n: int = 3) -> list[tuple[str, float]]:
    """For analyzers that only give a 1-best: pad with phonetically near phonemes at low confidence."""
    rest = max(0.0, 1.0 - confidence)
    out = [(normalize(best), confidence)]
    for q, d in nearest(best, n - 1):
        out.append((q, round(rest * (1 - d) / (n - 1), 4)))
    return out


# --- kana -> phonemes -------------------------------------------------------------------------

def _rows() -> dict[str, list[str]]:
    table: dict[str, list[str]] = {}
    gojuon = {
        "": "あいうえお", "k": "かきくけこ", "g": "がぎぐげご", "s": "さしすせそ", "z": "ざじずぜぞ",
        "t": "たちつてと", "d": "だぢづでど", "n": "なにぬねの", "h": "はひふへほ", "b": "ばびぶべぼ",
        "p": "ぱぴぷぺぽ", "m": "まみむめも", "r": "らりるれろ",
    }
    for cons, kana in gojuon.items():
        for k, v in zip(kana, VOWELS, strict=True):
            table[k] = ([cons] if cons else []) + [v]
    table.update({
        "し": ["sh", "i"], "じ": ["j", "i"], "ち": ["ch", "i"], "ぢ": ["j", "i"],
        "つ": ["ts", "u"], "づ": ["z", "u"], "ふ": ["f", "u"],
        "や": ["y", "a"], "ゆ": ["y", "u"], "よ": ["y", "o"],
        "わ": ["w", "a"], "を": ["o"], "ゐ": ["i"], "ゑ": ["e"],
        "ん": ["N"], "っ": ["cl"], "ゔ": ["v", "u"],
        "ぁ": ["a"], "ぃ": ["i"], "ぅ": ["u"], "ぇ": ["e"], "ぉ": ["o"],
        "ゃ": ["y", "a"], "ゅ": ["y", "u"], "ょ": ["y", "o"], "ゎ": ["w", "a"],
    })
    small = {"ゃ": "a", "ゅ": "u", "ょ": "o", "ぇ": "e"}
    for base, cons in {"き": "ky", "ぎ": "gy", "に": "ny", "ひ": "hy", "び": "by", "ぴ": "py",
                       "み": "my", "り": "ry", "し": "sh", "じ": "j", "ち": "ch", "ぢ": "j"}.items():
        for s, v in small.items():
            table[base + s] = [cons, v]
    table.update({
        "てぃ": ["t", "i"], "でぃ": ["d", "i"], "とぅ": ["t", "u"], "どぅ": ["d", "u"],
        "てゅ": ["ty", "u"], "でゅ": ["dy", "u"],
        "ふぁ": ["f", "a"], "ふぃ": ["f", "i"], "ふぇ": ["f", "e"], "ふぉ": ["f", "o"], "ふゅ": ["hy", "u"],
        "うぃ": ["w", "i"], "うぇ": ["w", "e"], "うぉ": ["w", "o"], "いぇ": ["y", "e"],
        "つぁ": ["ts", "a"], "つぃ": ["ts", "i"], "つぇ": ["ts", "e"], "つぉ": ["ts", "o"],
        "ゔぁ": ["v", "a"], "ゔぃ": ["v", "i"], "ゔぇ": ["v", "e"], "ゔぉ": ["v", "o"],
        "すぃ": ["s", "i"], "ずぃ": ["z", "i"],
    })
    return table


KANA: dict[str, list[str]] = _rows()


def to_hiragana(text: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def kana_to_morae(text: str) -> tuple[list[list[str]], list[str]]:
    """Split kana into morae (each a phoneme list). Returns (morae, unknown characters).
    "ー" repeats the previous vowel."""
    s = to_hiragana(text)
    morae: list[list[str]] = []
    unknown: list[str] = []
    i = 0
    while i < len(s):
        if s[i] == "ー":
            if morae and morae[-1][-1] in VOWELS + (MORAIC_NASAL,):
                morae.append([morae[-1][-1]])
            i += 1
            continue
        if s[i:i + 2] in KANA:
            morae.append(list(KANA[s[i:i + 2]]))
            i += 2
        elif s[i] in KANA:
            morae.append(list(KANA[s[i]]))
            i += 1
        else:
            if not s[i].isspace():
                unknown.append(s[i])
            i += 1
    return morae, unknown
