"""Phase 1 (lyrics mode): phoneme segments with top-3 candidates and confidences.

1. whisperX (large-v3) transcribes the source into timed utterances.
2. pyopenjtalk turns each utterance's text into phonemes.
3. A wav2vec2 phoneme-recognition CTC model (IPA output) gives a posteriorgram; the phoneme
   sequence is force-aligned onto it for boundaries.
4. Each phoneme segment's candidates are the phonemes with the most posterior mass over its
   frames (top 3, normalized to confidences) -- so the 1-best is what the acoustics say, which
   may differ from the transcript's phoneme.

Silence rules (as in melody mode): segments below the silence level are dropped, and vowel / "N"
segments that are mostly unvoiced (breath, whisper) are dropped too. Utterances whose alignment
is implausible (typically whisper hallucinating on music or noise) are skipped as a whole.

Needs the `lyrics` extra (torch, whisperX, pyopenjtalk). Imports are local so melody mode never
loads any of it.
"""

from __future__ import annotations

import gc
import json
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pyworld

from .phonemes import INVENTORY, is_voiced_sustained, normalize
from .progress import progress

SR = 16000
WHISPER_MODEL = "large-v3"
PHONEME_MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
PAD_SEC = 0.15                 # context around each utterance
MIN_ALIGN_PROB = 0.02          # geometric-mean token probability below which an utterance is skipped
MAX_SEG_SEC = {"vowel": 1.2, "consonant": 0.25}
MIN_VOICED_RATIO = 0.3         # sustained phonemes need at least this share of voiced frames

# pyopenjtalk phoneme -> IPA tokens of the phoneme model. The first token is the alignment target;
# all of them count toward the phoneme's posterior mass.
IPA: dict[str, list[str]] = {
    "a": ["a", "aː", "ɐ", "ä", "ɑ", "a."], "i": ["i", "iː", "ɪ", "i."], "u": ["ɯ", "u", "uː", "ɯᵝ", "ʊ", "ɨ"],
    "e": ["e", "eː", "e̞", "ɛ"], "o": ["o", "oː", "o̞", "ɔ"], "N": ["ɴ", "N", "ŋ", "n̩"],
    "k": ["k", "kʰ", "kh"], "g": ["ɡ", "ɣ"], "ky": ["kʲ", "c"], "gy": ["ɡʲ", "ɟ"],
    "s": ["s", "s̪"], "z": ["z"], "sh": ["ɕ", "ʃ"], "j": ["dʑ", "ʑ", "dʒ"],
    "t": ["t", "t̪", "tʰ"], "d": ["d"], "ty": ["tʲ"], "dy": ["dʲ"], "ts": ["ts"], "ch": ["tɕ", "tʃ"],
    "n": ["n"], "ny": ["ɲ", "nʲ"], "h": ["h"], "hy": ["ç"], "f": ["ɸ", "f"],
    "b": ["b", "β"], "by": ["bʲ"], "p": ["p", "pʰ"], "py": ["pʲ"], "m": ["m"], "my": ["mʲ"],
    "r": ["ɾ", "r", "ɽ"], "ry": ["rʲ"], "y": ["j"], "w": ["w"], "v": ["v"],
}
assert set(IPA) == set(INVENTORY)


@dataclass
class PhonemeSegment:
    start_sec: float
    end_sec: float
    phoneme: str                         # 1-best
    candidates: list[tuple[str, float]]  # top 3, best first
    f0_hz: float | None
    f0_std_cents: float
    rms_db: float


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
    progress.log(msg.strip())


def transcribe(audio: np.ndarray, device: str, batch_size: int = 8,
               model_name: str = WHISPER_MODEL) -> list[dict]:
    import os

    import torch
    import whisperx

    compute_type = "float16" if device == "cuda" else "int8"
    progress.stage(f"loading whisperX model ({model_name}, {device})")
    model = whisperx.load_model(model_name, device, compute_type=compute_type, language="ja",
                                threads=(os.cpu_count() or 4) if device == "cpu" else 4)
    # The voice activity detection over the whole file runs first and reports nothing.
    progress.stage("whisperX: voice detection, then transcription (%)", 100)
    result = model.transcribe(audio, batch_size=batch_size, language="ja", print_progress=True,
                              progress_callback=progress.update)
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return [s for s in result["segments"] if s.get("text", "").strip()]


class PhonemeModel:
    def __init__(self, device: str):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        self.torch = torch
        self.device = device
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(PHONEME_MODEL)
        self.model = Wav2Vec2ForCTC.from_pretrained(PHONEME_MODEL).to(device).eval()
        if device == "cuda":
            self.model = self.model.half()
        vocab = json.load(open(hf_hub_download(PHONEME_MODEL, "vocab.json"), encoding="utf-8"))
        self.blank = vocab["<pad>"]
        self.target_id = {ph: vocab[toks[0]] for ph, toks in IPA.items()}
        self.members = [np.array([vocab[t] for t in toks if t in vocab]) for toks in (IPA[p] for p in INVENTORY)]

    def posteriors(self, audio: np.ndarray) -> np.ndarray:
        """(frames, vocab) probabilities."""
        torch = self.torch
        inputs = self.extractor(audio, sampling_rate=SR, return_tensors="pt").input_values.to(self.device)
        if self.device == "cuda":
            inputs = inputs.half()
        with torch.inference_mode():
            logits = self.model(inputs).logits[0].float()
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def phoneme_mass(self, probs: np.ndarray) -> np.ndarray:
        """(frames, len(INVENTORY)): per frame, the best token probability of each phoneme."""
        return np.stack([probs[:, m].max(axis=1) for m in self.members], axis=1)

    def align(self, probs: np.ndarray, phonemes: list[str]) -> list[tuple[int, int, float]] | None:
        """Forced alignment -> per phoneme (first frame, last frame + 1, token probability)."""
        torch = self.torch
        import torchaudio.functional as F

        targets = torch.tensor([[self.target_id[p] for p in phonemes]], dtype=torch.int32)
        log_probs = torch.from_numpy(np.log(probs + 1e-10))[None].float()
        try:
            ali, scores = F.forced_align(log_probs, targets, blank=self.blank)
        except RuntimeError:
            return None  # audio too short for the transcript
        spans = F.merge_tokens(ali[0], scores[0].exp())
        if len(spans) != len(phonemes):
            return None
        return [(s.start, s.end, float(s.score)) for s in spans]


def _g2p(text: str) -> list[str]:
    import pyopenjtalk

    out = []
    for ph in pyopenjtalk.g2p(text).split():
        ph = normalize(ph)
        if ph in INVENTORY:  # pau / cl are silence: left to the CTC blank
            out.append(ph)
    return out


def analyze(audio: np.ndarray, device: str | None = None,
            model_name: str = WHISPER_MODEL) -> list[PhonemeSegment]:
    import torch

    warnings.filterwarnings("ignore", message="(?s).*torchcodec")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"  transcribing with whisperX {model_name} on {device}")
    utterances = transcribe(audio, device, model_name=model_name)
    _log(f"  {len(utterances)} utterances; aligning phonemes with {PHONEME_MODEL}")
    progress.stage("loading phoneme model")
    model = PhonemeModel(device)
    progress.stage("phoneme alignment (utterances)", len(utterances))

    all_rms = []
    raw: list[tuple[PhonemeSegment, str, float]] = []  # (segment, transcript phoneme, voiced ratio)
    skipped = 0
    for ui, utt in enumerate(utterances):
        phonemes = _g2p(utt["text"])
        if not phonemes:
            continue
        a0 = max(0, int((utt["start"] - PAD_SEC) * SR))
        a1 = min(len(audio), int((utt["end"] + PAD_SEC) * SR))
        chunk = audio[a0:a1]
        if len(chunk) < SR * 0.1:
            continue
        probs = model.posteriors(chunk)
        spans = model.align(probs, phonemes)
        if spans is None or np.exp(np.mean(np.log([max(p, 1e-6) for *_, p in spans]))) < MIN_ALIGN_PROB:
            skipped += 1
            continue
        frame_sec = len(chunk) / SR / len(probs)
        mass = model.phoneme_mass(probs)
        cum = np.vstack([np.zeros(mass.shape[1]), np.cumsum(mass, axis=0)])

        x = chunk.astype(np.float64)
        f0, f0_t = pyworld.dio(x, SR, frame_period=5.0, f0_floor=60.0, f0_ceil=1100.0)
        f0 = pyworld.stonemask(x, f0, f0_t, SR)

        for pi, (ph, (s, e, _)) in enumerate(zip(phonemes, spans, strict=True)):
            # A phoneme lasts from its first frame to the next phoneme's first frame, capped.
            end = spans[pi + 1][0] if pi + 1 < len(spans) else e + 1
            cap = MAX_SEG_SEC["vowel" if is_voiced_sustained(ph) else "consonant"]
            end = min(end, s + max(1, int(round(cap / frame_sec))), len(probs))
            if end <= s:
                continue
            m = (cum[end] - cum[s]) / (end - s)
            order = np.argsort(m)[::-1][:3]
            conf = m[order] / max(m.sum(), 1e-9)
            cands = [(INVENTORY[j], round(float(c), 4)) for j, c in zip(order, conf, strict=True)]

            t0, t1 = s * frame_sec, end * frame_sec
            seg_f0 = f0[int(t0 * 200): max(int(t0 * 200) + 1, int(t1 * 200))]
            voiced = seg_f0[seg_f0 > 0]
            samples = x[int(t0 * SR): int(t1 * SR)]
            rms = 20 * np.log10(np.sqrt(np.mean(samples ** 2)) + 1e-12) if samples.size else -120.0
            voiced_ratio = voiced.size / max(1, seg_f0.size)
            if voiced.size >= 3:
                med = float(np.median(voiced))
                std = float(np.std(1200 * np.log2(voiced / med)))
            else:
                med, std = None, 0.0
            raw.append((PhonemeSegment(a0 / SR + t0, a0 / SR + t1, cands[0][0], cands, med, std, float(rms)),
                        ph, voiced_ratio))
            all_rms.append(rms)
        progress.update(ui + 1)
        if (ui + 1) % 200 == 0:
            _log(f"  aligned {ui + 1}/{len(utterances)} utterances")

    if not raw:
        return []
    loud = float(np.percentile(all_rms, 95))
    threshold = max(-45.0, loud - 35.0)
    kept = [
        seg for seg, label, voiced_ratio in raw
        if seg.rms_db > threshold
        # a vowel / N (by transcript or by acoustics) that is barely voiced is breath or whisper
        and not ((is_voiced_sustained(label) or is_voiced_sustained(seg.phoneme))
                 and (voiced_ratio < MIN_VOICED_RATIO or seg.f0_hz is None))
    ]
    _log(f"  {len(kept)} phoneme segments kept of {len(raw)} "
         f"({skipped} utterances skipped as implausible)")
    return kept

