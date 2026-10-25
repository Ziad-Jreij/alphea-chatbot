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
    """Returns fixed scores per candidate label (default 0.05)."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def __call__(self, text, candidate_labels, hypothesis_template, multi_label, **kwargs):
        self.calls += 1
        ordered = sorted(candidate_labels, key=lambda l: self.scores.get(l, 0.05), reverse=True)
        return {"labels": ordered, "scores": [self.scores.get(l, 0.05) for l in ordered]}


SAFETY = crit.RISK_CATEGORIES["high"][0]
DELIVERY = crit.RISK_CATEGORIES["medium"][0]
QUESTION = crit.RISK_CATEGORIES["low"][0]


@pytest.fixture
def fake_model(monkeypatch):
    def install(scores):
        pipe = FakeZeroShot(scores)
        monkeypatch.setattr(crit, "ensure_crit_pipe", lambda *a, **k: pipe)
        return pipe
    return install


def urgency(low, medium, high):
    return {"low urgency": low, "medium urgency": medium, "high urgency": high}


def test_risk_is_the_default_strategy():
    assert crit.CRITICALITY_STRATEGY == "risk"


def test_high_keyword_short_circuits_model(fake_model):
    pipe = fake_model({QUESTION: 0.9})
    res = crit.infer_criticality_for_text("someone stole my credit card", "neutral")
    assert res["criticality_level"] == "high"
    assert pipe.calls == 0


# Risk strategy

def test_risk_category_escalates_and_is_explained(fake_model):
    fake_model({SAFETY: 0.92, DELIVERY: 0.6})
    res = crit.infer_criticality_for_text("the lamp wiring is exposed", "neutral")
    assert res["criticality_level"] == "high"
    assert res["risk_category"] == SAFETY
    assert SAFETY in res["explanation"]
    assert res["source"] == "risk_model"


def test_problem_without_risk_is_medium(fake_model):
    fake_model({SAFETY: 0.2, DELIVERY: 0.8, QUESTION: 0.3})
    res = crit.infer_criticality_for_text("my parcel is late", "neutral")
    assert res["criticality_level"] == "medium"
    assert res["escalation"] == "potential"


def test_routine_question_is_low(fake_model):
    fake_model({SAFETY: 0.05, DELIVERY: 0.2, QUESTION: 0.9})
    res = crit.infer_criticality_for_text("what are your opening hours", "neutral")
    assert res["criticality_level"] == "low"
    assert res["escalation"] == "none"


def test_risk_just_below_threshold_is_not_high(fake_model):
    fake_model({SAFETY: crit.RISK_HIGH_THRESHOLD - 0.05, DELIVERY: 0.8})
    res = crit.infer_criticality_for_text("the charger feels warm", "neutral")
    assert res["criticality_level"] == "medium"


def test_medium_keyword_is_a_floor_not_a_cap(fake_model):
    # Regression: "charged twice" (medium keyword) used to return MEDIUM before
    # the model ran, so a genuine fraud/urgent signal could never reach HIGH.
    fake_model({crit.RISK_CATEGORIES["high"][2]: 0.95})
    res = crit.infer_criticality_for_text("my card was charged twice and I need this fixed immediately", None)
    assert res["criticality_level"] == "high"

    fake_model({QUESTION: 0.9})
    res = crit.infer_criticality_for_text("I think I was charged twice?", None)
    assert res["criticality_level"] == "medium"
    assert res["source"].startswith("keyword_medium")
    assert res["matched_keyword"] == "charged twice"


@pytest.mark.parametrize("raw_label", ["annoyance", "nervousness"])
def test_raw_go_emotions_labels_apply_the_prior(fake_model, raw_label):
    # Regression: raw labels the action server passes ("annoyance") had no
    # entry in EMOTION_WEIGHTS, so the prior was silently skipped.
    fake_model({SAFETY: 0.6, DELIVERY: 0.6})
    calm = crit.infer_criticality_for_text("the heater smells odd", "neutral")
    upset = crit.infer_criticality_for_text("the heater smells odd", raw_label)
    assert upset["distribution"]["high"] > calm["distribution"]["high"]


def test_structural_rule_overrides_low_risk_model(fake_model):
    fake_model({QUESTION: 0.9})
    res = crit.infer_criticality_for_text("the battery caught fire", "neutral")
    assert res["criticality_level"] == "high"
    assert "+safety_rule" in res["source"]


def test_risk_distribution_is_a_distribution(fake_model):
    fake_model({SAFETY: 0.4, DELIVERY: 0.7, QUESTION: 0.3})
    res = crit.infer_criticality_for_text("my order is a bit late", "disappointed")
    assert sum(res["distribution"].values()) == pytest.approx(1.0)
    assert res["reason_code"] and res["explanation"]


# Legacy urgency strategy (CRITICALITY_STRATEGY=urgency)

def test_urgency_low_stays_low(fake_model):
    fake_model(urgency(0.85, 0.10, 0.05))
    res = crit.infer_criticality_for_text("what are your store hours", "neutral", strategy="urgency")
    assert res["criticality_level"] == "low"
    assert res["source"] == "model"


def test_urgency_high_above_threshold(fake_model):
    fake_model(urgency(0.10, 0.15, 0.75))
    res = crit.infer_criticality_for_text("I need this fixed today", None, strategy="urgency")
    assert res["criticality_level"] == "high"


def test_urgency_emotion_prior_raises_urgency(fake_model):
    fake_model(urgency(0.45, 0.40, 0.15))
    calm = crit.infer_criticality_for_text("my package has not arrived", "satisfied", strategy="urgency")
    upset = crit.infer_criticality_for_text("my package has not arrived", "anxious", strategy="urgency")
    assert upset["distribution"]["high"] > calm["distribution"]["high"]


# ── Finance-rule negation guard ──────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "there was no unauthorized charge, I just want a copy of the invoice",
    "I heard this might be a scam but I'm just checking",
])
def test_negated_finance_rule_does_not_fire(text):
    level, suffix, _ = structural(text)
    assert "+finance_rule" not in suffix


def test_safety_rule_ignores_negation_words():
    # Deliberate: a missed safety escalation is worse than a false alarm.
    level, suffix, _ = structural("I can't believe it caught fire")
    assert suffix == "+safety_rule"


@pytest.mark.parametrize("text", [
    "I have contacted you three times about a missing refund",
    "this is the fourth time I'm writing",
    "six emails and still no reply",
    "I called support 5 times",
])
def test_repeated_contact_is_frustration(text):
    _, suffix, _ = structural(text)
    assert "+frustration_rule" in suffix


@pytest.mark.parametrize("text", [
    "it costs three times the price of the old one",
    "I ordered this four times before and loved it",
    "I need two emails changed on my account",
])
def test_repeat_lookalikes_are_not_frustration(text):
    _, suffix, _ = structural(text)
    assert "+frustration_rule" not in suffix
