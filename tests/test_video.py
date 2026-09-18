"""Video layout: layer roles, panel geometry, random slots, and the chroma key colour."""

import numpy as np
import pytest

from madgen.cli import build_parser
from madgen.render import RenderedVoice, build_layers
from madgen.target import PERCUSSION_CHANNEL, TargetUnit, Voice
from madgen.video import Span, lead_box, panel_slots, pick_key_color, split_spans_over_slots


def _voice(track, name, *, lyrics=False, channel=0, notes=4, dur=1.0):
    voice = Voice(track, name, 0, origin="ust" if lyrics else "midi", channel=channel)
    voice.units = [TargetUnit(i, i * 2.0, dur, 440.0, 100, note_index=i) for i in range(notes)]
    return voice


def _rendered(voice):
    from madgen.video import VideoNote

    notes = [VideoNote(u.start_sec, u.duration_sec, "src.mp4", 0.0) for u in voice.units]
    return RenderedVoice(voice, np.zeros(1, dtype=np.float32), notes)


def _args(*extra):
    return build_parser().parse_args(["render", "--db", "x", "--melody", "y", "--out", "z.wav", *extra])


def test_layers_put_drums_behind_and_vocals_in_front():
    voices = [
        _rendered(_voice(1, "Standard", channel=PERCUSSION_CHANNEL, notes=8)),
        _rendered(_voice(2, "Chord", notes=4)),
        _rendered(_voice(3, "Arpeggio", notes=3)),
        _rendered(_voice(4, "Track1", lyrics=True, notes=6)),
    ]
    layers = build_layers(voices, 20.0, _args())
    assert [lyr.name.split(":")[0] for lyr in layers] == ["background", "panel0", "panel1", "lead"]
    background, lead = layers[0], layers[-1]
    assert (background.x, background.y, background.width, background.height) == (0, 0, 1280, 720)
    assert not background.transparent           # the drums cover the canvas
    assert lead.width < 1280 and lead.x > 0     # the sung part is centred and smaller
    for panel in layers[1:-1]:
        assert panel.width < lead.width
        assert panel.y + panel.height <= lead.y or panel.y >= lead.y + lead.height


def test_layer_option_overrides_roles():
    voices = [_rendered(_voice(1, "Standard", channel=PERCUSSION_CHANNEL)), _rendered(_voice(2, "Chord"))]
    layers = build_layers(voices, 20.0, _args("--layer", "Standard=lead", "--layer", "Chord=background"))
    assert layers[0].name == "background" and not layers[0].transparent
    assert layers[-1].name == "lead"
    with pytest.raises(SystemExit):
        build_layers(voices, 20.0, _args("--layer", "Nope=lead"))
    with pytest.raises(SystemExit):
        build_layers(voices, 20.0, _args("--layer", "Chord=middle"))


def test_without_drums_the_longest_sounding_track_is_the_background():
    voices = [_rendered(_voice(1, "Short", notes=2)), _rendered(_voice(2, "Long", notes=9)),
              _rendered(_voice(3, "Main", notes=4))]
    layers = build_layers(voices, 30.0, _args())
    assert layers[0].name == "background"
    # "Main" leads by name, "Long" sounds longest so it goes to the back.
    assert [lyr.name for lyr in layers][-1] == "lead"


def test_panel_layouts():
    voices = [_rendered(_voice(1, "Drums", channel=PERCUSSION_CHANNEL))]
    voices += [_rendered(_voice(i, f"P{i}", notes=3)) for i in range(2, 12)]
    voices.append(_rendered(_voice(20, "Vox", lyrics=True)))
    fixed = build_layers(voices, 30.0, _args("--panel-layout", "fixed"))
    auto = build_layers(voices, 30.0, _args("--panel-layout", "auto"))
    random_ = build_layers(voices, 30.0, _args("--panel-layout", "random"))
    assert len(fixed) == 1 + 6 + 1          # 10 parts share 6 slots
    assert len(auto) == 1 + 10 + 1          # one slot per part
    assert len(random_) == 1 + 9 + 1        # a fixed pool of random slots
    assert all(lyr.width > 0 and lyr.height > 0 for lyr in fixed + auto + random_)


def test_panel_slots_stay_clear_of_the_lead():
    lead = lead_box(1280, 720)
    for count in (1, 3, 6, 9, 12):
        for x, y, w, h in panel_slots(count, 1280, 720, lead):
            assert 0 <= x and x + w <= 1280
            assert 0 <= y and y + h <= 720
            assert y + h <= lead[1] or y >= lead[1] + lead[3]


def test_split_spans_over_slots_keeps_the_timeline():
    spans = [Span(0, 10, "a.mp4", 1.0), Span(10, 5, None), Span(15, 10, "b.mp4", 2.0)]
    slots = split_spans_over_slots(spans, 3, seed=1)
    assert len(slots) == 3
    for slot in slots:
        assert sum(s.frames for s in slot) == 25          # every slot covers the whole timeline
    shown = [s.video_ref for slot in slots for s in slot if s.video_ref]
    assert sorted(shown) == ["a.mp4", "b.mp4"]            # each clip is placed exactly once


def test_pick_key_color_avoids_colours_in_the_material():
    green = np.zeros((2, 8, 8, 3), np.uint8)
    green[..., 1] = 250
    assert pick_key_color(green) != "green"
    magenta = np.zeros((2, 8, 8, 3), np.uint8)
    magenta[..., 0] = magenta[..., 2] = 250
    assert pick_key_color(magenta) != "magenta"
    assert pick_key_color(np.zeros((0, 8, 8, 3), np.uint8)) == "magenta"   # no material: a default


def test_repeated_segment_plays_on_instead_of_restarting():
    from madgen.video import VideoNote, build_spans

    # Four notes in a row that all picked the same segment (drums do this).
    same = [VideoNote(i * 1.0, 1.0, "src.mp4", 100.0) for i in range(4)]
    spans = build_spans([same], 4.0, fps=30)
    assert len(spans) == 1                      # one continuous span, not four restarts
    assert spans[0].frames == 120 and spans[0].source_start == 100.0

    # Different segments still get their own spans.
    varied = [VideoNote(i * 1.0, 1.0, "src.mp4", 100.0 + i) for i in range(3)]
    assert len(build_spans([varied], 3.0, fps=30)) == 3
