"""Per-track filters and EQ.

The whole track is filtered in one go in the frequency domain: the magnitude response of the
requested filters is evaluated at the FFT bins and multiplied in. That keeps the dependency list
to numpy alone, and being zero-phase it does not smear transients the way a steep time-domain
filter would -- which matters here, where the material is full of short percussive hits.

Specs are written as `type:args`, several separated by commas:

    hp:80              cut below 80 Hz (12 dB/oct; hp:80:4 for 24 dB/oct)
    lp:8000            cut above 8 kHz
    peak:3000:+4       lift 3 kHz by 4 dB (peak:3000:+4:2 to narrow it; Q defaults to 1)
    lowshelf:200:-3    everything below 200 Hz down 3 dB
    highshelf:5000:+2  everything above 5 kHz up 2 dB
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_ORDER = 8
MAX_GAIN_DB = 24.0


@dataclass(frozen=True)
class Filter:
    kind: str
    freq: float
    gain_db: float = 0.0
    q: float = 1.0
    order: int = 2

    def response(self, f: np.ndarray) -> np.ndarray:
        """Magnitude response at the frequencies `f` (Hz)."""
        # Guard the DC bin: every shape below divides by the frequency somewhere.
        x = np.maximum(f, 1e-6) / self.freq
        if self.kind == "hp":
            return np.sqrt(x ** (2 * self.order) / (1 + x ** (2 * self.order)))
        if self.kind == "lp":
            return np.sqrt(1 / (1 + x ** (2 * self.order)))
        gain = 10 ** (self.gain_db / 20)
        if self.kind == "peak":
            # 1 at the edges, `gain` at the centre, width set by Q.
            bell = 1 / (1 + (self.q * (x - 1 / x)) ** 2)
            return 1 + (gain - 1) * bell
        if self.kind in ("lowshelf", "highshelf"):
            # A smooth step from one level to the other, centred on the frequency.
            step = 1 / (1 + x ** (2 * self.order))     # 1 below the frequency, 0 above
            if self.kind == "highshelf":
                step = 1 - step
            return 1 + (gain - 1) * step
        raise ValueError(f"unknown filter {self.kind!r}")


def parse(spec: str) -> list[Filter]:
    """`"hp:80,peak:3000:+4"` -> the filters it names."""
    out = []
    for part in spec.split(","):
        fields = part.strip().split(":")
        kind = fields[0].strip().lower()
        try:
            if kind in ("hp", "lp"):
                freq = float(fields[1])
                order = int(fields[2]) if len(fields) > 2 else 2
                if not 1 <= order <= MAX_ORDER:
                    raise ValueError(f"order must be within 1..{MAX_ORDER}")
                out.append(Filter(kind, freq, order=order))
            elif kind in ("peak", "lowshelf", "highshelf"):
                freq, gain = float(fields[1]), float(fields[2])
                q = float(fields[3]) if len(fields) > 3 else 1.0
                if abs(gain) > MAX_GAIN_DB:
                    raise ValueError(f"gain must be within +-{MAX_GAIN_DB:g} dB")
                if q <= 0:
                    raise ValueError("Q must be positive")
                out.append(Filter(kind, freq, gain, q))
            else:
                raise ValueError(f"unknown filter type {kind!r}")
        except (IndexError, ValueError) as e:
            raise SystemExit(
                f"--filter {part.strip()!r}: {e}. Write it as hp:80, lp:8000, peak:3000:+4[:Q], "
                f"lowshelf:200:-3 or highshelf:5000:+2, several separated by commas"
            ) from None
        if out[-1].freq <= 0:
            raise SystemExit(f"--filter {part.strip()!r}: the frequency must be above 0 Hz")
    return out


def apply(audio: np.ndarray, filters: list[Filter], sample_rate: int) -> np.ndarray:
    if not filters or audio.size == 0:
        return audio
    spectrum = np.fft.rfft(audio.astype(np.float64))
    freqs = np.fft.rfftfreq(audio.size, 1 / sample_rate)
    response = np.ones_like(freqs)
    for f in filters:
        response *= f.response(freqs)
    return np.fft.irfft(spectrum * response, n=audio.size).astype(np.float32)
