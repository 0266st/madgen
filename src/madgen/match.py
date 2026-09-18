"""Phase 3: unit selection -- pick the lowest total-cost segment sequence with Viterbi DP.

Melody mode: total = sum(target cost + pitch cost) + sum(concatenation cost), a weighted sum.
Lyrics mode: the same DP over a lexicographic cost (see the lyrics section below).

Candidates are first narrowed to the top K per unit with a vectorized numpy scan over the
whole corpus (faiss is not needed at this corpus size), then the DP runs over K x K.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .db import Corpus
from .target import TargetUnit, hz_to_midi


@dataclass
class CostWeights:
    pitch_near: float = 0.15      # per semitone, within `near_semitones` (pitch shift fixes it)
    pitch_far: float = 1.0        # per semitone beyond that
    near_semitones: float = 1.0
    coverage: float = 1.5         # segment shorter than the note -> the rest is silence
    stability: float = 0.5        # per 100 cents of pitch wobble inside the segment
    loudness: float = 0.3         # per 10 dB below the corpus' loud level
    join: float = 0.4             # neighbours that were not adjacent in the source
    repeat: float = 0.2           # the same segment twice in a row
    legato_gap_sec: float = 0.05  # notes further apart than this are not "connected"
    top_k: int = 40


def _target_costs(unit: TargetUnit, corpus: Corpus, seg_semis: np.ndarray, seg_dur: np.ndarray,
                  loud_ref: float, w: CostWeights) -> np.ndarray:
    diff = np.abs(12 * np.log2(unit.target_f0_hz / 440.0) - seg_semis)
    pitch = np.where(
        diff <= w.near_semitones,
        w.pitch_near * diff,
        w.pitch_near * w.near_semitones + w.pitch_far * (diff - w.near_semitones),
    )
    coverage = np.minimum(seg_dur, unit.duration_sec) / unit.duration_sec
    cost = (
        pitch
        + w.coverage * (1 - coverage)
        + w.stability * corpus.f0_std / 100
        + w.loudness * np.maximum(0.0, loud_ref - corpus.rms_db) / 10
    )
    return cost


def select_units(units: list[TargetUnit], corpus: Corpus, w: CostWeights | None = None) -> np.ndarray:
    """Return, for each target unit, the index (into `corpus` arrays) of the chosen segment."""
    w = w or CostWeights()
    n = len(units)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    k = min(w.top_k, len(corpus))
    seg_semis = 12 * np.log2(corpus.f0 / 440.0)
    seg_dur = corpus.end - corpus.start
    loud_ref = float(np.percentile(corpus.rms_db, 90))

    cand = np.empty((n, k), dtype=np.int64)
    local = np.empty((n, k), dtype=np.float64)
    cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    for i, u in enumerate(units):
        key = (round(u.target_f0_hz, 2), round(u.duration_sec, 2), u.phoneme)
        if key not in cache:
            c = _target_costs(u, corpus, seg_semis, seg_dur, loud_ref, w)
            top = np.argpartition(c, k - 1)[:k] if k < len(c) else np.arange(len(c))
            cache[key] = (top, c[top])
        cand[i], local[i] = cache[key]

    # Viterbi.
    score = local[0].copy()
    back = np.zeros((n, k), dtype=np.int64)
    for i in range(1, n):
        prev, cur = cand[i - 1], cand[i]
        gap = units[i].start_sec - (units[i - 1].start_sec + units[i - 1].duration_sec)
        if gap > w.legato_gap_sec:
            trans = np.zeros((k, k))
        else:
            follows = corpus.next_idx[prev][:, None] == cur[None, :]
            same = prev[:, None] == cur[None, :]
            trans = np.where(follows, 0.0, np.where(same, w.repeat, w.join))
        total = score[:, None] + trans
        back[i] = np.argmin(total, axis=0)
        score = total[back[i], np.arange(k)] + local[i]

    path = np.empty(n, dtype=np.int64)
    j = int(np.argmin(score))
    for i in range(n - 1, -1, -1):
        path[i] = cand[i, j]
        j = back[i, j]
    return path


# --- lyrics mode ------------------------------------------------------------------------------
#
# Candidates are compared lexicographically: (phoneme cost, pitch + length cost, concatenation
# cost). Each tier is a small integer, and the tiers are packed into one int64 with place values
# large enough that no sum of a lower tier over the whole path can outweigh one step of a higher
# tier -- so a plain Viterbi sum over the packed values *is* the lexicographic comparison.

@dataclass
class LyricsWeights:
    # tier 1: phoneme. 1-best match < 2nd < 3rd < no match (phonetic distance); within a rank,
    # lower confidence costs more, in `confidence_steps` buckets (so ties -- and tie-breaks -- happen).
    rank_band: int = 10
    confidence_steps: int = 4
    distance_steps: int = 10
    # tier 2: pitch (in steps of `pitch_step_cents` beyond the no-correction threshold) + length
    # shortfall (in tenths of the note that would be silence)
    pitch_step_cents: float = 100.0
    pitch_free_cents: float = 25.0
    pitch_max_steps: int = 24
    coverage_steps: int = 10
    # tier 3: 0 if the neighbours were adjacent in the source, else 1
    legato_gap_sec: float = 0.05
    top_k: int = 50


def phoneme_cost(target: str, cand_phoneme: np.ndarray, cand_conf: np.ndarray, w: LyricsWeights) -> np.ndarray:
    """Tier-1 cost for every segment (vectorized over rows of the top-3 candidate table)."""
    from .phonemes import distance

    n = len(cand_phoneme)
    cost = np.full(n, -1, dtype=np.int64)
    for rank in range(3):
        hit = (cand_phoneme[:, rank] == target) & (cost < 0)
        bucket = np.floor((1.0 - cand_conf[:, rank]) * w.confidence_steps).clip(0, w.confidence_steps - 1)
        cost[hit] = rank * w.rank_band + bucket[hit].astype(np.int64)
    miss = cost < 0
    if miss.any():
        best = cand_phoneme[miss, 0]
        dist = np.array([distance(target, b) if b else 1.0 for b in best])
        cost[miss] = 3 * w.rank_band + np.round(dist * w.distance_steps).astype(np.int64)
    return cost


def _lower_tier(unit: TargetUnit, corpus: Corpus, idx: np.ndarray, use_pitch: bool, w: LyricsWeights) -> np.ndarray:
    seg_dur = corpus.end[idx] - corpus.start[idx]
    coverage = np.minimum(seg_dur, unit.duration_sec) / unit.duration_sec
    cost = np.round((1 - coverage) * w.coverage_steps).astype(np.int64)
    if use_pitch:
        f0 = corpus.f0[idx]
        cents = np.where(np.isnan(f0), np.inf, np.abs(1200 * np.log2(unit.target_f0_hz / np.nan_to_num(f0, nan=1.0))))
        steps = np.ceil(np.maximum(0.0, cents - w.pitch_free_cents) / w.pitch_step_cents)
        cost += np.minimum(steps, w.pitch_max_steps).astype(np.int64)
    return cost


def select_units_lyrics(units: list[TargetUnit], corpus: Corpus, w: LyricsWeights | None = None) -> np.ndarray:
    from .phonemes import is_voiced_sustained

    w = w or LyricsWeights()
    n = len(units)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if corpus.cand_phoneme is None or corpus.cand_conf is None:
        raise ValueError("lyrics mode needs a phoneme corpus")
    k = min(w.top_k, len(corpus))

    # Early cut-off: segments whose 1-best is the target phoneme, per phoneme.
    by_best: dict[str, np.ndarray] = {}
    for ph in set(u.phoneme for u in units):
        by_best[ph] = np.nonzero(corpus.cand_phoneme[:, 0] == ph)[0]
    phon_cache: dict[str, np.ndarray] = {}

    max_tier2 = w.pitch_max_steps + w.coverage_steps
    tier3_scale = 1                       # tier-3 values are 0/1
    tier2_scale = n * tier3_scale + 1     # > any path sum of tier 3
    tier1_scale = n * max_tier2 * tier2_scale + tier2_scale  # > any path sum of tiers 2+3

    cand = np.empty((n, k), dtype=np.int64)
    local = np.empty((n, k), dtype=np.int64)
    for i, u in enumerate(units):
        use_pitch = is_voiced_sustained(u.phoneme)
        best_hits = by_best[u.phoneme]
        if len(best_hits) >= k:
            # Enough 1-best matches: the 2nd/3rd candidates cannot win, skip them.
            idx = best_hits
            t1 = phoneme_cost(u.phoneme, corpus.cand_phoneme[idx], corpus.cand_conf[idx], w)
        else:
            if u.phoneme not in phon_cache:
                phon_cache[u.phoneme] = phoneme_cost(u.phoneme, corpus.cand_phoneme, corpus.cand_conf, w)
            idx = np.arange(len(corpus))
            t1 = phon_cache[u.phoneme]
        t2 = _lower_tier(u, corpus, idx, use_pitch, w)
        packed = t1 * tier1_scale + t2 * tier2_scale
        take = np.argpartition(packed, k - 1)[:k] if k < len(packed) else np.arange(len(packed))
        if len(take) < k:  # tiny corpus: repeat the best to fill the table
            take = np.resize(take[np.argsort(packed[take])], k)
        cand[i] = idx[take]
        local[i] = packed[take]

    score = local[0].copy()
    back = np.zeros((n, k), dtype=np.int64)
    for i in range(1, n):
        prev, cur = cand[i - 1], cand[i]
        gap = units[i].start_sec - (units[i - 1].start_sec + units[i - 1].duration_sec)
        if gap > w.legato_gap_sec:
            trans = np.zeros((k, k), dtype=np.int64)
        else:
            follows = corpus.next_idx[prev][:, None] == cur[None, :]
            trans = np.where(follows, 0, tier3_scale).astype(np.int64)
        total = score[:, None] + trans
        back[i] = np.argmin(total, axis=0)
        score = total[back[i], np.arange(k)] + local[i]

    path = np.empty(n, dtype=np.int64)
    j = int(np.argmin(score))
    for i in range(n - 1, -1, -1):
        path[i] = cand[i, j]
        j = back[i, j]
    return path


# --- percussion -------------------------------------------------------------------------------

def _pick_drum_sample(drum, corpus: Corpus, taken: list[int],
                      materials: dict[str, tuple[str, ...]] | None) -> int:
    """The segment this instrument will play, out of everything the corpus holds."""
    seg_dur = corpus.end - corpus.start
    loud_ref = float(np.percentile(corpus.rms_db, 90))
    phonemes = (materials or {}).get(drum.name, drum.phonemes)
    best, best_score = None, None
    for rank, phoneme in enumerate(phonemes):
        hits = np.nonzero(corpus.cand_phoneme[:, 0] == phoneme)[0]
        if hits.size == 0:
            continue
        voiced = ~np.isnan(corpus.f0[hits])
        score = (
            # The instrument's phonemes are listed best first, and each step down costs 4 -- about
            # as much as a segment twice too long. So a later phoneme wins when the material at the
            # head of the list is poor, but the order still decides between equally good segments.
            rank * 4.0
            # Another instrument already sounds like this.
            + np.where(np.isin(hits, taken) if taken else False, 8.0, 0.0)
            # A hit wants a segment about as long as itself: much longer gets cut anyway,
            # much shorter leaves a gap.
            + 3.0 * np.abs(seg_dur[hits] - drum.max_sec) / drum.max_sec
            + 1.5 * np.maximum(0.0, loud_ref - corpus.rms_db[hits]) / 10
        )
        if drum.voiced is True:
            # A kick or a tom needs a pitched, low sound.
            score += np.where(voiced, 0.0, 6.0)
            low = np.where(voiced, np.nan_to_num(corpus.f0[hits], nan=999.0), 999.0)
            score += np.clip((low - 120.0) / 120.0, 0.0, 4.0)
        elif drum.voiced is False:
            score += np.where(voiced, 3.0, 0.0)   # prefer noise over a pitched sound
        j = int(hits[int(np.argmin(score))])
        if best_score is None or score.min() < best_score:
            best, best_score = j, float(score.min())
    if best is None:  # nothing resembling the instrument: fall back to the shortest loud bit
        best = int(np.argmin(3.0 * seg_dur + np.maximum(0.0, loud_ref - corpus.rms_db) / 10))
    return best


def choose_drum_samples(units: list[TargetUnit], corpus: Corpus, samples: dict[str, int],
                        materials: dict[str, tuple[str, ...]] | None = None) -> dict[str, int]:
    """Fill in `samples`: one segment per instrument the units call for, the way a kit is built.

    Whoever picks first gets the better material, because everyone after pays the reuse penalty,
    so the order is fixed by the instrument's importance (Drum.priority) rather than by whichever
    one the song happens to start with. Otherwise the kick of a song opening on a hi-hat would be
    built from the leftovers, and splitting a track into voices differently would change the kit.

    Instruments are keyed by name, not by note: General MIDI gives one instrument several numbers
    (a kick is both 35 and 36), and a kit with two different kicks sounds wrong. Pass one dict
    through every drum voice of a render so the kit stays the same throughout.
    """
    from .phonemes import drum_for

    if corpus.cand_phoneme is None:
        raise ValueError("percussion sampling needs a phoneme corpus")
    wanted: dict[str, object] = {}
    for unit in units:
        drum = drum_for(hz_to_midi(unit.target_f0_hz))
        wanted.setdefault(drum.name, drum)
    for name, drum in sorted(wanted.items(), key=lambda kv: (kv[1].priority, kv[0])):
        if name not in samples:
            samples[name] = _pick_drum_sample(drum, corpus, list(samples.values()), materials)
    return samples


def select_units_percussion(units: list[TargetUnit], corpus: Corpus,
                            samples: dict[str, int] | None = None,
                            materials: dict[str, tuple[str, ...]] | None = None) -> np.ndarray:
    """One segment per drum instrument, reused for every hit, the way a sampler works.

    A drum track's notes name instruments rather than pitches, so nothing here looks at f0 as a
    target. What matters is the material's character: the phonemes listed for the instrument, a
    length close to what the hit needs, and enough level to cut through.
    """
    from .phonemes import drum_for

    chosen = {} if samples is None else samples
    choose_drum_samples(units, corpus, chosen, materials)
    return np.array([chosen[drum_for(hz_to_midi(u.target_f0_hz)).name] for u in units],
                    dtype=np.int64)
