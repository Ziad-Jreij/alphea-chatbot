"""Emotion bucketing and smoothing (no model needed)."""
import pytest

from components import emotion

GO_EMOTIONS_LABELS = {
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
}


def test_every_go_emotions_label_is_mapped():
    assert set(emotion._RAW_TO_BUSINESS) == GO_EMOTIONS_LABELS
    assert set(emotion._RAW_TO_BUSINESS.values()) <= set(emotion.BUSINESS_EMOTION_DISPLAY)


@pytest.mark.parametrize("raw,bucket", [
    ("annoyance", "frustrated"),
    ("fear", "anxious"),
    ("sadness", "disappointed"),
    ("gratitude", "satisfied"),
    ("confusion", "confused"),
    ("neutral", "neutral"),
    ("ANGER", "frustrated"),        # case-insensitive
])
def test_map_to_business_emotion(raw, bucket):
    assert emotion.map_to_business_emotion(raw) == bucket


@pytest.mark.parametrize("raw", ["", None, "not-a-label"])
def test_unknown_labels_default_to_neutral(raw):
    assert emotion.map_to_business_emotion(raw) == "neutral"


@pytest.mark.parametrize("raw,polarity", [
    ("anger", "negative"),
    ("joy", "positive"),
    ("confusion", "ambiguous"),
    ("neutral", "neutral"),
])
def test_coarse_polarity(raw, polarity):
    assert emotion.coarse_from_label(raw) == polarity


def test_single_outlier_is_smoothed_away():
    label, reason = emotion.smooth_emotion("neutral", ["frustrated", "frustrated", "frustrated"])
    assert label == "frustrated"
    assert reason == "smoothed_from_neutral_to_frustrated"


def test_first_turn_is_a_new_signal():
    assert emotion.smooth_emotion("frustrated", []) == ("frustrated", "new_signal")


def test_repeated_label_is_stable():
    assert emotion.smooth_emotion("satisfied", ["satisfied"]) == ("satisfied", "stable")


def test_window_only_looks_back_four_turns():
    # Old frustration outside the window must not outvote recent satisfaction.
    history = ["frustrated"] * 10 + ["satisfied", "satisfied"]
    label, _ = emotion.smooth_emotion("satisfied", history)
    assert label == "satisfied"


def test_most_recent_turn_wins_a_tie():
    label, _ = emotion.smooth_emotion("anxious", ["frustrated"])
    assert label == "anxious"
