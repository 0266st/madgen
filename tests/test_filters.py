"""Filters and EQ: what each spec does to the spectrum, and how bad specs are reported."""

import numpy as np
import pytest

from madgen import filters

SR = 44100


def _tone_mix(freqs, seconds=1.0):
    t = np.arange(int(SR * seconds)) / SR
    return sum(np.sin(2 * np.pi * f * t) for f in freqs).astype(np.float32)


def _level_db(audio, freq):
    spectrum = np.fft.rfft(audio.astype(np.float64))
    bin_ = int(round(freq * audio.size / SR))
    return 20 * np.log10(abs(spectrum[bin_]) / (audio.size / 2) + 1e-12)


def _change(spec, freqs):
    before = _tone_mix(freqs)
    after = filters.apply(before, filters.parse(spec), SR)
    assert after.size == before.size
    return {f: _level_db(after, f) - _level_db(before, f) for f in freqs}


def test_highpass_and_lowpass():
    hp = _change("hp:200", [50, 200, 1000])
    assert hp[50] < -18          # 2 octaves below the corner, at 12 dB/oct
    assert -4 < hp[200] < -2     # the corner itself sits at -3 dB
    assert abs(hp[1000]) < 0.5

    lp = _change("lp:2000", [1000, 10000])
    assert abs(lp[1000]) < 1
    assert lp[10000] < -20

    steep = _change("hp:200:4", [50])
    assert steep[50] < hp[50] - 12   # a higher order cuts harder


def test_peak_and_shelves():
    peak = _change("peak:3000:+6", [200, 3000, 10000])
    assert 5.5 < peak[3000] < 6.5
    assert abs(peak[200]) < 0.5 and abs(peak[10000]) < 1.5

    narrow = _change("peak:3000:+6:4", [1500, 3000])
    wide = _change("peak:3000:+6:0.5", [1500, 3000])
    assert narrow[1500] < wide[1500]     # a higher Q leaves the neighbours alone

    low = _change("lowshelf:200:-6", [50, 2000])
    assert -6.5 < low[50] < -5.5 and abs(low[2000]) < 0.5
    high = _change("highshelf:5000:+6", [200, 15000])
    assert 5 < high[15000] < 6.5 and abs(high[200]) < 0.5


def test_several_filters_at_once():
    both = _change("hp:100,peak:1000:-6", [50, 1000, 5000])
    assert both[50] < -10 and -6.5 < both[1000] < -5.5 and abs(both[5000]) < 0.5


def test_no_filters_leaves_the_audio_alone():
    audio = _tone_mix([440], 0.1)
    assert filters.apply(audio, [], SR) is audio
    assert filters.apply(np.zeros(0, dtype=np.float32), filters.parse("hp:80"), SR).size == 0


@pytest.mark.parametrize("spec", ["", "hp", "hp:0", "hp:-5", "wobble:100", "peak:1000",
                                  "peak:1000:+40", "peak:1000:+4:0", "hp:100:99"])
def test_bad_specs_are_refused(spec):
    with pytest.raises(SystemExit):
        filters.parse(spec)
