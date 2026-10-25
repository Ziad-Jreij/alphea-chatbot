"""Criticality guardrails: keyword overrides, structural rules, model fusion.

The zero-shot model is replaced by a fake pipeline, so these tests pin down the
deterministic layers that decide most real escalations.
"""
import pytest

from components import criticality as crit

LOW_DIST = {"low": 0.6, "medium": 0.3, "high": 0.1}


def structural(text):
    dist, suffix, pattern = crit._apply_structural_rules(text, dict(LOW_DIST))
    return max(dist, key=dist.get), suffix, pattern


# ── Keyword overrides (data/criticality_keywords.yml) ────────────────────────

@pytest.mark.parametrize("text", [
    "someone stole my credit card",
    "my account was hacked",
    "delete my account and all my data",
])
def test_high_keyword_forces_high(text):
    res = crit._apply_keyword_overrides_text(text)
    assert res["criticality_level"] == "high"
    assert res["escalation"] == "escalate_now"
    assert res["source"] == "keyword_high"
    assert res["reason_code"].startswith("keyword_high:")


@pytest.mark.parametrize("text", [
    "my card was not stolen",           # negation
    "I heard my account was hacked?",   # hearsay
])
def test_negated_or_hearsay_keyword_is_ignored(text):
    assert crit._apply_keyword_overrides_text(text) is None


def test_no_keyword_for_routine_message():
    assert crit._apply_keyword_overrides_text("where is my order") is None
    assert crit._apply_keyword_overrides_text("") is None


# ── Structural rules ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,suffix", [
    ("the charger caught fire", "+safety_rule"),
    ("smoke is coming out of the laptop", "+safety_rule"),
    ("the charger is smoking", "+safety_rule"),
    ("I got the wrong medicine", "+medical_rule"),
    ("the medicine is expired", "+medical_rule"),
    ("there is an unauthorized charge on my card", "+finance_rule"),
])
def test_hard_overrides_force_high(text, suffix):
    level, got_suffix, _ = structural(text)
    assert level == "high"
    assert got_suffix == suffix


@pytest.mark.parametrize("text", [
    "my coupon expired yesterday",      # was HIGH via `expired (...)?`
    "I don't smoke",                    # was HIGH via `smok(e|ing)`
    "thanks for the help",
    "how long does shipping take?",
])
def test_routine_messages_are_not_escalated(text):
    level, suffix, _ = structural(text)
    assert level == "low"
    assert suffix == ""


@pytest.mark.parametrize("text", [
    "my phone won't turn on",           # straight apostrophe
    "my phone won’t turn on",           # curly apostrophe (mobile keyboards)
    "the app keeps crashing",
])
def test_core_function_failure_bumps_to_high(text):
    level, suffix, _ = structural(text)
    assert level == "high"
    assert "+core_rule" in suffix


def test_contrast_clause_is_detected():
    _, suffix, _ = structural("it looks nice but the screen keeps freezing")
    assert "+contrast_rule" in suffix


def test_frustration_rule():
    level, suffix, _ = structural("I reported this three times and no one is helping me")
    assert level == "high"
    assert "+frustration_rule" in suffix


def test_boost_level_prob_renormalises():
    out = crit._boost_level_prob({"low": 0.6, "medium": 0.3, "high": 0.1}, "high", 0.9)
    assert out["high"] == pytest.approx(0.9)
    assert sum(out.values()) == pytest.approx(1.0)
    assert out["low"] > out["medium"]   # relative order of the rest is kept


# ── Full pipeline with a fake zero-shot model ────────────────────────────────

class FakeZeroShot:
    def __init__(self, low, medium, high):
        self.scores = {"low urgency": low, "medium urgency": medium, "high urgency": high}
        self.calls = 0

    def __call__(self, text, candidate_labels, hypothesis_template, multi_label):
        self.calls += 1
        ordered = sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)
        return {"labels": [k for k, _ in ordered], "scores": [v for _, v in ordered]}


@pytest.fixture
def fake_model(monkeypatch):
    def install(low, medium, high):
        pipe = FakeZeroShot(low, medium, high)
        monkeypatch.setattr(crit, "ensure_crit_pipe", lambda *a, **k: pipe)
        return pipe
    return install


def test_keyword_short_circuits_model(fake_model):
    pipe = fake_model(0.9, 0.05, 0.05)
    res = crit.infer_criticality_for_text("someone stole my credit card", "neutral")
    assert res["criticality_level"] == "high"
    assert pipe.calls == 0


def test_model_low_stays_low(fake_model):
    fake_model(0.85, 0.10, 0.05)
    res = crit.infer_criticality_for_text("what are your store hours", "neutral")
    assert res["criticality_level"] == "low"
    assert res["escalation"] == "none"
    assert res["source"] == "model"


def test_model_high_above_threshold(fake_model):
    fake_model(0.10, 0.15, 0.75)
    res = crit.infer_criticality_for_text("I need this fixed today", None)
    assert res["criticality_level"] == "high"


def test_emotion_prior_raises_urgency(fake_model):
    fake_model(0.45, 0.40, 0.15)
    calm = crit.infer_criticality_for_text("my package has not arrived", "satisfied")
    upset = crit.infer_criticality_for_text("my package has not arrived", "anxious")
    assert upset["distribution"]["high"] > calm["distribution"]["high"]


def test_structural_rule_applies_after_model(fake_model):
    fake_model(0.80, 0.15, 0.05)
    res = crit.infer_criticality_for_text("the battery caught fire", "neutral")
    assert res["criticality_level"] == "high"
    assert "+safety_rule" in res["source"]


def test_every_result_is_explainable(fake_model):
    fake_model(0.2, 0.6, 0.2)
    res = crit.infer_criticality_for_text("my order is a bit late", "disappointed")
    assert res["reason_code"]
    assert res["explanation"]
    assert set(res["distribution"]) == {"low", "medium", "high"}
