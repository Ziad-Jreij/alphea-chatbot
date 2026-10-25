"""ABSA exact-match path (embedding model disabled)."""
import pytest

from components import absa


@pytest.fixture(autouse=True)
def no_embeddings(monkeypatch):
    monkeypatch.setattr(absa, "_EMBEDDING_AVAILABLE", False)


def extract(text, emotion_label=None):
    return [
        (r["aspect"], r["aspect_sentiment"])
        for r in absa.extract_aspect_opinions(text, emotion_label=emotion_label)
    ]


def test_two_negative_aspects():
    assert sorted(extract("the delivery was late and the support was rude")) == [
        ("customer support", "negative"),
        ("delivery & shipping", "negative"),
    ]


def test_single_positive_aspect():
    assert extract("great product quality") == [("product quality", "positive")]


@pytest.mark.parametrize("emotion_label", [None, "annoyance", "joy"])
def test_contrast_sentence_pairs_each_aspect_with_its_own_opinion(emotion_label):
    # Regression: "fast" was also an aspect term (performance), so shipping got
    # no opinion and fell back to the user's emotion.
    got = extract("the price is too high but shipping was fast", emotion_label)
    assert sorted(got) == [("delivery & shipping", "positive"), ("price", "negative")]


def test_opinion_word_is_not_a_separate_aspect():
    assert extract("the app is fast") == [("app experience", "positive")]


def test_opinion_word_alone_still_counts_as_aspect():
    # With no other aspect named, "slow" is the performance complaint itself.
    assert extract("it is so slow") == [("performance", "negative")]


def test_negation_flips_opinion():
    assert extract("the support was not helpful") == [("customer support", "negative")]


@pytest.mark.parametrize("text", ["hello", "", "/cancel_order", "what time is it"])
def test_no_aspects_for_non_feedback(text):
    assert extract(text) == []


def test_result_schema():
    (row,) = absa.extract_aspect_opinions("great product quality")
    assert set(row) >= {
        "aspect", "aspect_term", "aspect_category", "business_aspect",
        "business_aspect_display", "opinion_term", "aspect_sentiment",
        "aspect_sentiment_confidence", "aspect_match_type", "aspect_similarity",
    }
    assert row["business_aspect"] in absa.BUSINESS_ASPECTS
    assert 0.0 <= row["aspect_sentiment_confidence"] <= 1.0


def test_word_boundaries_respected():
    # "app" must not match inside "happy".
    assert all(aspect != "app experience" for aspect, _ in extract("I am happy with it overall, great job"))
