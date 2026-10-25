"""Custom actions: order-ID handling around forms (regressions from Phase 2)."""
import pytest
from rasa_sdk import Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from actions import actions


@pytest.fixture(autouse=True)
def no_models(monkeypatch):
    # Validators score emotion/criticality on every turn; not under test here.
    monkeypatch.setattr(actions, "_infer_emotion_slots", lambda tracker: {})


def make_tracker(text="", entities=(), slots=None):
    return Tracker(
        sender_id="test",
        slots=dict(slots or {}),
        latest_message={"text": text, "intent": {"name": "x"}, "entities": list(entities)},
        events=[],
        paused=False,
        followup_action=None,
        active_loop={},
        latest_action_name=None,
    )


def order_entity(value):
    return {"entity": "order_id", "value": value}


def slot_values(events):
    return {e["name"]: e["value"] for e in events if e.get("event") == "slot"}


# ── action_clear_context ─────────────────────────────────────────────────────

def test_clear_context_wipes_stale_order_id():
    tracker = make_tracker("I want to cancel an order", slots={"order_id": "ORD-55555"})
    events = actions.ActionClearContext().run(CollectingDispatcher(), tracker, {})
    assert slot_values(events)["order_id"] is None


def test_clear_context_keeps_order_id_from_current_message():
    tracker = make_tracker("cancel ord-12345", [order_entity("ord-12345")], {"order_id": "ORD-55555"})
    events = actions.ActionClearContext().run(CollectingDispatcher(), tracker, {})
    assert events[-1] == SlotSet("order_id", "ORD-12345")


def test_clear_context_ignores_invalid_entity():
    tracker = make_tracker("cancel 1234", [order_entity("1234")])
    events = actions.ActionClearContext().run(CollectingDispatcher(), tracker, {})
    assert slot_values(events)["order_id"] is None


# ── Pre-filled order_id in every order form ──────────────────────────────────

FORM_VALIDATORS = [
    actions.ValidateOrderStatusForm,
    actions.ValidateReturnForm,
    actions.ValidateCancelOrderForm,
    actions.ValidateDamageReportForm,
    actions.ValidateAddressUpdateForm,
]


@pytest.mark.parametrize("validator", FORM_VALIDATORS)
@pytest.mark.parametrize("prefilled,expected", [
    ("ORD-12345", "ORD-12345"),
    ("ord-12345", "ORD-12345"),
    ("1234", None),          # invalid NLU entity → form asks for it normally
])
def test_prefilled_order_id_is_normalised(validator, prefilled, expected):
    tracker = make_tracker(slots={"order_id": prefilled, "requested_slot": None})
    out = validator().validate_order_id(prefilled, CollectingDispatcher(), tracker, {})
    assert out["order_id"] == expected


# ── Conversation-level risk ──────────────────────────────────────────────────

def run_turns(turns):
    """Feed (level, confidence) turns through _update_conversation_risk."""
    slots = {}
    state = {}
    for level, confidence in turns:
        state = actions._update_conversation_risk(make_tracker("msg", slots=slots), level, confidence)
        slots.update(state)
    return state


def test_confident_low_turn_does_not_escalate():
    # Regression: a LOW turn with 0.80 confidence was read as 0.80 urgency and
    # escalated "I want to cancel order ORD-98765" to a human.
    state = run_turns([("low", 0.80)])
    assert state["conversation_criticality"] == "low"
    assert state["conversation_escalation"] == "none"


def test_single_medium_turn_is_potential_not_high():
    state = run_turns([("low", 0.9), ("medium", 0.95)])
    assert state["conversation_escalation"] != "escalate_now"


def test_sustained_medium_becomes_potential():
    state = run_turns([("medium", 0.6)] * 3)
    assert state["conversation_criticality"] == "medium"
    assert state["conversation_escalation"] == "potential"


def test_any_high_turn_escalates():
    state = run_turns([("low", 0.9), ("high", 0.66)])
    assert state["conversation_escalation"] == "escalate_now"


def test_early_high_fades_after_calm_turns():
    state = run_turns([("high", 0.9)] + [("low", 0.9)] * 4)
    assert state["false_trigger_candidate"] is True
    assert state["conversation_escalation"] != "escalate_now"


def test_every_turn_is_audited():
    state = run_turns([("low", 0.8), ("medium", 0.6)])
    assert [e["turn_level"] for e in state["escalation_audit_log"]] == ["low", "medium"]
