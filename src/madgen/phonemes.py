"""Phoneme inventory (pyopenjtalk's), kana -> phonemes, and the phonetic distance table.

Kana conversion is a plain table so that rendering from a UST/USTX needs no G2P engine;
it produces the same symbols pyopenjtalk.g2p() does for kana.
"""

from __future__ import annotations

from dataclasses import dataclass
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


# --- percussion -------------------------------------------------------------------------------
#
# A drum track's note numbers name instruments, not pitches (General MIDI channel 10), so the
# usual pitch matching is meaningless there. Instead each instrument borrows a phoneme that sounds
# like it: the unvoiced consonants of speech are short bursts of noise, which is what a hi-hat or a
# snare is, and the voiced plosives are the low thumps a kick needs.

@dataclass(frozen=True)
class Drum:
    name: str
    phonemes: tuple[str, ...]   # preferred material, best first
    max_sec: float              # hits are cut to this, however long the note is
    level_db: float = 0.0       # relative to the other parts
    voiced: bool | None = None  # True: needs a pitched (low) sound, False: wants noise


_HAT = Drum("ハイハット", ("ts", "ch", "t", "k", "s"), 0.08, -2.0, voiced=False)
_SNARE = Drum("スネア", ("sh", "s", "ts", "ch"), 0.15, 0.0, voiced=False)
_KICK = Drum("キック", ("b", "d", "g", "m"), 0.20, 1.0, voiced=True)
_TOM = Drum("タム", ("d", "b", "g"), 0.20, 0.0, voiced=True)
_CYMBAL = Drum("シンバル", ("sh", "s"), 0.60, -3.0, voiced=False)
def _perc(name: str, phonemes: tuple[str, ...] = ("t", "k", "d", "ts"), max_sec: float = 0.15) -> Drum:
    return Drum(name, phonemes, max_sec, -1.0)


_PERC = _perc("パーカッション")
_SHAKER = Drum("シェイカー", ("sh", "s", "ts"), 0.10, -4.0, voiced=False)

# General MIDI percussion key map (the entries a typical track actually uses).
DRUMS: dict[int, Drum] = {
    35: _KICK, 36: _KICK,
    37: Drum("リムショット", ("t", "k", "p"), 0.08, -2.0, voiced=False),
    38: _SNARE, 40: _SNARE,
    39: Drum("クラップ", ("p", "t", "k"), 0.12, 0.0, voiced=False),
    41: _TOM, 43: _TOM, 45: _TOM, 47: _TOM, 48: _TOM, 50: _TOM,
    42: _HAT, 44: _HAT,
    46: Drum("オープンハイハット", ("sh", "s", "ts"), 0.20, -2.0, voiced=False),
    49: _CYMBAL, 52: _CYMBAL, 55: _CYMBAL, 57: _CYMBAL,
    51: Drum("ライド", ("ch", "ts", "k"), 0.20, -3.0, voiced=False),
    53: Drum("ライドベル", ("ch", "ts"), 0.15, -3.0, voiced=False),
    59: Drum("ライド", ("ch", "ts", "k"), 0.20, -3.0, voiced=False),
    54: Drum("タンバリン", ("ts", "ch", "sh"), 0.12, -3.0, voiced=False),
    56: Drum("カウベル", ("k", "t"), 0.12, -2.0),
    # Each of these is its own instrument, and a kit that plays them all with one sound loses the
    # pattern they make together; the high/low pair of an instrument shares a sound on purpose.
    60: _perc("ハイボンゴ"), 61: _perc("ローボンゴ"),
    62: _perc("ハイコンガ"), 63: _perc("ハイコンガ"), 64: _perc("ローコンガ"),
    65: _perc("ハイティンバレ"), 66: _perc("ローティンバレ"),
    67: _perc("ハイアゴゴ", ("k", "t", "ts")), 68: _perc("ローアゴゴ", ("k", "t", "ts")),
    69: _SHAKER, 70: _SHAKER, 82: _SHAKER,
    75: Drum("クラベス", ("k", "t"), 0.08, -2.0, voiced=False),
}
DEFAULT_DRUM = _PERC


def drum_for(note: int) -> Drum:
    return DRUMS.get(note, DEFAULT_DRUM)
