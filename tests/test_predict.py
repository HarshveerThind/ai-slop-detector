import math

import pytest

from predict import reasons_for, verdict_for

NAN = math.nan
BAND = [0.2, 0.5]


@pytest.mark.parametrize("overall, verdict", [
    (0.9, "likely_ai"),
    (0.51, "likely_ai"),
    (0.5, "unsure"),
    (0.3, "unsure"),
    (0.2, "unsure"),
    (0.19, "likely_human"),
])
def test_verdict_uses_the_unsure_band(overall, verdict):
    assert verdict_for(overall, BAND) == verdict


def test_reasons_come_in_order_when_true():
    assert reasons_for(0.7, 0.9, 0.7, True) == [
        "Synthetic sounding voice",
        "AI generated visuals",
        "Creator disclosed AI content",
        "Upload patterns common in AI channels",
    ]


def test_no_reasons_below_cutoff_or_when_missing():
    assert reasons_for(0.69, 0.69, 0.69, False) == []
    assert reasons_for(NAN, NAN, 0.1, False) == []
