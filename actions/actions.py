from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
import re
from rasa_sdk.events import SlotSet, EventType
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import torch
import logging
from threading import Lock
from components.emotion import ensure_emotion_pipe, coarse_from_label, top_emotion 
from components.criticality import infer_criticality_for_text

_init_lock = Lock()
# Fake database for demo
ORDER_DB = {
    "ORD-12345": {"status": "Shipped", "eta": "2 days"},
    "ORD-98765": {"status": "Processing", "eta": "4 days"},
}

ORDER_RE = re.compile(r"^ORD-\d{5}$", re.IGNORECASE)
EMAIL_RE = re.compile(r"[^@]+@[^@]+\.[^@]+")

logger = logging.getLogger(__name__)

_CRIT2SCORE = {"low": 0.2, "medium": 0.6, "high": 0.9}
_SCORE2CRIT = lambda s: ("high" if s>=0.75 else "medium" if s>=0.45 else "low")
_ESCALATION_MAP = {"low":"none","medium":"potential","high":"escalate_now"}
def _coarse(emotion: str) -> str:
    return coarse_from_label(emotion)
def _ensure_pipe():  # keep your existing name to minimize diff
    return ensure_emotion_pipe()
def clear_context_slots() -> List[Dict[Text, Any]]:
    """Clear slots that shouldn't leak across flows."""
    return [SlotSet("order_id", None), SlotSet("email", None), SlotSet("issue_description", None)]
def normalize_order_id(s: str) -> Text | None:
    s = (s or "").strip().upper().replace(" ", "")
    if s.startswith("ORD") and "-" not in s and len(s) == 8 and s[3:].isdigit():
        s = f"ORD-{s[3:]}"
    return s if ORDER_RE.fullmatch(s or "") else None
def _infer_emotion_slots(tracker):
    text = (tracker.latest_message.get("text") or "").strip()
    if not text:
        return {}
    pipe = _ensure_pipe()
    preds = pipe(text)[0]
    top = max(preds, key=lambda x: x["score"])
    top_label = top["label"]
    sentiment = _coarse(top_label)
    agg = _merge_emotion_aggregate(tracker, top_label)
    overall = agg.get("overall_emotion")
    logger.info(
        f"[form-emotion] text='{text}' -> top={top_label}, sentiment={sentiment}, overall={overall}"
    )
    return {"top_emotion": top_label, "sentiment": sentiment, **agg}
def _infer_criticality_slots(text: str, emotion: str) -> Dict[Text, Any]:
    text = (text or "").strip()
    if not text:
        return {}

    result = infer_criticality_for_text(
        text,
        emotion,
        # keywords default as per component
    )

    level = result["criticality_level"]
    score = float(result["criticality_score"])
    escalation = result["escalation"]
    dist = result["distribution"]

    logger.info(
        "[crit] text=%r emo=%s -> level=%s score=%.2f escalation=%s dist=%s",
        text,
        (emotion or "none"),
        level,
        score,
        escalation,
        {k: f"{v:.2f}" for k, v in dist.items()},
    )

    return {
        "criticality_level": level,
        "criticality_score": score,
        "escalation": escalation,
    }

def _best_overall_from_tally(tally: Dict[str, int], last_top: str) -> str:
    if not tally:
        return last_top or "neutral"
    # max count; break ties in favor of the most recent top label
    maxc = max(tally.values())
    candidates = [e for e, c in tally.items() if c == maxc]
    return last_top if last_top in candidates else sorted(candidates)[0]

def _merge_emotion_aggregate(tracker, new_label):
    tally = tracker.get_slot("emotion_tally") or {}
    history = tracker.get_slot("emotion_history") or []

    tally[new_label] = tally.get(new_label, 0) + 1

    history.append(new_label)
    if len(history) > 10:
        history.pop(0)

    weights = {}
    for i, e in enumerate(reversed(history)):
        weights[e] = weights.get(e, 0) + (1 / (i + 1))

    overall = max(weights.items(), key=lambda x: x[1])[0]
    return {
        "emotion_tally": tally,
        "emotion_history": history,
        "overall_emotion": overall,
    }
class ActionUpdateConversationRisk(Action):
    def name(self) -> str: 
        return "action_update_conversation_risk"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        print("ACTION RISK UPDATE STARTED SUCCESSFULLY!")

        # 1) TURN-LEVEL CRITICALITY via shared helper
        text = (tracker.latest_message.get("text") or "").strip()
        # use latest top_emotion if already set, but ActionInferEmotion can pass a fresher one;
        # for now we can still fall back to the slot:
        emo_for_prior = tracker.get_slot("top_emotion")
        crit = _infer_criticality_slots(text, emo_for_prior)
        level = (crit.get("criticality_level") or "low").lower()
        score = float(crit.get("criticality_score") or _CRIT2SCORE.get(level, 0.2))
        turn_escalation = crit.get("escalation") or _ESCALATION_MAP.get(level, "none")

        # 2) HISTORY & TALLY (same as before)
        history = list(tracker.get_slot("criticality_history") or [])
        tally: Dict[str, int] = dict(tracker.get_slot("criticality_tally") or {})

        history.append({"level": level, "score": score})
        history = history[-15:]

        tally[level] = tally.get(level, 0) + 1

        # 3) CONVERSATION-LEVEL SCORE
        w = 1.0
        total_w = 0.0
        agg = 0.0
        for item in reversed(history):
            agg += w * float(item["score"])
            total_w += w
            w *= 0.85
        conv_score = agg / max(total_w, 1e-9)
        conv_level = _SCORE2CRIT(conv_score)
        conv_escalation = _ESCALATION_MAP[conv_level]

        # guardrail: any high in history → force escalate_now
        if any(h["level"] == "high" for h in history):
            conv_escalation = "escalate_now"
            conv_level = "high"
            conv_score = max(conv_score, 0.9)

        # 4) LOGGING
        text = (tracker.latest_message.get("text") or "").strip()
        emo = (tracker.get_slot("top_emotion") or "unknown")
        sentiment = (tracker.get_slot("sentiment") or "ambiguous")

        logger.info(
            "[conv-risk] conv=%s text=%r emo=%s sentiment=%s | "
            "turn_level=%s turn_score=%.2f turn_escalation=%s | "
            "conv_level=%s conv_score=%.2f conv_escalation=%s history=%s",
            tracker.sender_id,
            text,
            emo,
            sentiment,
            level,
            score,
            turn_escalation,
            conv_level,
            conv_score,
            conv_escalation,
            [h["level"] for h in history][-5:],
        )

        return [
            # per-turn
            SlotSet("criticality_level", level),
            SlotSet("criticality_score", score),
            SlotSet("escalation", turn_escalation),
            # conversation-level
            SlotSet("criticality_history", history),
            SlotSet("criticality_tally", tally),
            SlotSet("conversation_criticality", conv_level),
            SlotSet("conversation_criticality_score", conv_score),
            SlotSet("conversation_escalation", conv_escalation),
        ]

class ActionEscalateNow(Action):
    def name(self) -> Text:
        return "action_escalate_now"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        # Here you can create a ticket, ping a webhook, or notify a live queue
        # Example message to the user:
        dispatcher.utter_message(
            text="I’m escalating this to a human specialist right now. You’ll be connected shortly."
        )
        # Add your integration here, e.g. requests.post(...)
        return []

class ActionOfferHandoff(Action):
    def name(self) -> Text:
        return "action_offer_handoff"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        # Gentle handoff prompt
        dispatcher.utter_message(
            text="This looks a bit urgent. Would you like me to bring a human agent into the chat?"
        )
        return []
class ActionInferEmotion(Action):
    def name(self) -> Text:
        return "action_infer_emotion"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        text = (tracker.latest_message.get("text") or "").strip()
        if not text:
            return []

        pipe = _ensure_pipe()
        preds = pipe(text)[0]
        top = max(preds, key=lambda x: x["score"])
        top_label = top["label"]
        sentiment = _coarse(top_label)

        top3 = sorted(
            ((p["label"], float(p["score"])) for p in preds),
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        agg = _merge_emotion_aggregate(tracker, top_label)
        overall = agg.get("overall_emotion")

        logger.info(
            "[emotion] text=%r -> top_emotion=%s sentiment=%s top3=%s overall=%s",
            text,
            top_label,
            sentiment,
            top3,
            overall,
        )

        events: List[EventType] = [
            SlotSet("top_emotion", top_label),
            SlotSet("sentiment", sentiment),
            SlotSet("emotion_tally", agg["emotion_tally"]),
            SlotSet("overall_emotion", overall),
        ]

        # Also update conversation risk (criticality + escalation)
        risk_action = ActionUpdateConversationRisk()
        risk_events = risk_action.run(dispatcher, tracker, domain)
        events.extend(risk_events)

        return events
class ActionSoftAffirmation(Action):
    def name(self) -> str:
        return "action_soft_affirmation"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        # prevent back-to-back soft prompts within 20 seconds
        last = tracker.get_slot("last_soft_affirmation_ts")
        now = datetime.utcnow()
        if last:
            try:
                last_dt = datetime.fromisoformat(last)
                if (now - last_dt) < timedelta(seconds=20):
                    # silently skip repeating the same menu
                    return []
            except Exception:
                pass

        dispatcher.utter_message(
            text="Got it—just to be sure, what would you like to do?",
            buttons=[
                {"title": "Track my order", "payload": "/ask_order_status"},
                {"title": "Report damaged item", "payload": "/report_damaged_product"},
                {"title": "Reset password", "payload": "/reset_password"},
                {"title": "Talk to a human", "payload": "/talk_to_human"},
            ],
        )
        return [SlotSet("last_soft_affirmation_ts", now.isoformat())]
class ValidateDamageReportForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_damage_report_form"

    def validate_order_id(
        self, value: Text, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> Dict[Text, Any]:
        out = _infer_emotion_slots(tracker)  # ← always add emotion
        # Only validate when this slot is being asked
        if tracker.get_slot("requested_slot") != "order_id":
            out["order_id"] = tracker.get_slot("order_id")
            return out

        if not value:
            out["order_id"] = None
            return out

        v = normalize_order_id(value)
        if v and ORDER_RE.fullmatch(v):
            out["order_id"] = v
            return out

        dispatcher.utter_message(response="utter_order_id_invalid")
        out["order_id"] = None
        return out

    def validate_issue_description(
        self, value: Text, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> Dict[Text, Any]:
        out = _infer_emotion_slots(tracker)  # ← always add emotion
        if tracker.get_slot("requested_slot") != "issue_description":
            out["issue_description"] = tracker.get_slot("issue_description")
            return out

        if value and len(value.strip()) >= 5:
            out["issue_description"] = value.strip()
            return out

        dispatcher.utter_message(text="Please add a few more details about the damage.")
        out["issue_description"] = None
        return out
class ActionCancelOrder(Action):
    def name(self) -> Text: return "action_cancel_order"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        oid = tracker.get_slot("order_id"); confirm = tracker.get_slot("confirm_cancel")
        if not oid: 
            dispatcher.utter_message(text="I need the order ID to cancel.")
            return []
        if not confirm:
            dispatcher.utter_message(text="Okay, I’ll keep the order as is.")
            return clear_context_slots()
        rec = ORDER_DB.get(oid)
        if not rec:
            dispatcher.utter_message(text=f"Order {oid} not found.")
            return clear_context_slots()
        # business rule: allow cancel within 24h and if not shipped
        created = datetime.fromisoformat(rec.get("created_at", "2025-10-01"))
        if rec["status"].lower() == "shipped" or datetime.now() - created > timedelta(hours=24):
            dispatcher.utter_message(text=f"Order {oid} can’t be canceled now. You can start a return after delivery.")
            return clear_context_slots()
        dispatcher.utter_message(text=f"Order {oid} has been canceled and refunded. You’ll see the refund in 3–5 days.")
        return clear_context_slots()

class ActionStartReturn(Action):
    def name(self) -> Text: return "action_start_return"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        oid = tracker.get_slot("order_id"); reason = tracker.get_slot("return_reason")
        if not oid or not reason:
            dispatcher.utter_message(text="I’m missing details for the return.")
            return []
        if oid not in ORDER_DB:
            dispatcher.utter_message(text=f"Order {oid} not found.")
            return clear_context_slots()
        dispatcher.utter_message(text=f"Return initiated for {oid} (reason: {reason}). You’ll receive a label by email shortly.")
        return clear_context_slots()
# --- Actions ---
class ActionLookupOrderStatus(Action):
    def name(self) -> Text:
        return "action_lookup_order_status"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        order_id = tracker.get_slot("order_id")
        if not order_id:
            dispatcher.utter_message(text="I couldn’t find an order ID. Please try again.")
            return []
        rec = ORDER_DB.get(order_id)
        if not rec:
            dispatcher.utter_message(text=f"Sorry, I can’t find {order_id}. Double-check the ID (e.g., ORD-12345).")
        else:
            dispatcher.utter_message(text=f"Order {order_id}: {rec['status']} — ETA {rec['eta']}.")
        return []

class ActionPasswordReset(Action):
    def name(self) -> Text:
        return "action_password_reset"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        email = tracker.get_slot("email")
        if not email:
            dispatcher.utter_message(text="I need the account email to send a reset.")
            return []
        dispatcher.utter_message(text=f"Got it. A password reset email has been sent to {email}.")
        # clear context so future turns don't get hijacked
        return [SlotSet("email", None)]

class ActionRequestHandoff(Action):
    def name(self) -> Text: 
        return "action_request_handoff"

    def run(self, dispatcher, tracker, domain) -> List[EventType]:
        overall = tracker.get_slot("overall_emotion") or "unknown"
        last = tracker.get_slot("top_emotion") or "unknown"

        # internal log for ops/agent UI pickup
        logger.info(
            f"[handoff] ticket created | overall={overall} last={last} | conv_id={tracker.sender_id}"
        )

        # optional: mark that a handoff was requested
        return [SlotSet("handoff_requested", True)]

# Optional: clear context when user starts a new flow
class ActionClearContext(Action):
    def name(self) -> Text:
        return "action_clear_context"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        return clear_context_slots() + [SlotSet("emotion_tally", None), SlotSet("overall_emotion", None)]


# --- Validators ---
class ValidateOrderStatusForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_order_status_form"

    def validate_order_id(self, value, dispatcher, tracker, domain) -> Dict[Text, Any]:
        out = _infer_emotion_slots(tracker)  # ← add emotion every turn
        # Don't validate if not currently requesting this slot
        if tracker.get_slot("requested_slot") != "order_id":
            out["order_id"] = tracker.get_slot("order_id")
            return out

        if not value:
            out["order_id"] = None
            return out

        v = (value or "").strip().upper().replace(" ", "")
        if re.fullmatch(r"ORD-\d{5}", v):
            out["order_id"] = v
            return out

        dispatcher.utter_message(response="utter_order_id_invalid")
        out["order_id"] = None
        return out

class ValidatePasswordResetForm(FormValidationAction):
    def name(self) -> Text: return "validate_password_reset_form"

    def validate_email(self, value: Text, dispatcher, tracker, domain) -> Dict[Text, Any]:
        out = _infer_emotion_slots(tracker)
        if tracker.get_slot("requested_slot") != "email":
            out["email"] = tracker.get_slot("email")
            return out
        if value and EMAIL_RE.fullmatch(value.strip()):
            out["email"] = value.strip()
            return out
        dispatcher.utter_message(response="utter_email_invalid")
        out["email"] = None
        return out
class ActionRecordDamageTicket(Action):
    def name(self) -> Text:
        return "action_record_damage_ticket"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        order_id = tracker.get_slot("order_id")
        desc = tracker.get_slot("issue_description")
        if not order_id or not desc:
            dispatcher.utter_message(text="I’m missing some details. Please provide the order ID and a short description.")
            return []
        # Here you’d create a real ticket (Jira/Email/DB). We just confirm:
        dispatcher.utter_message(text=f"Ticket created for {order_id}: '{desc}'. A support agent will follow up shortly.")
        # Clear context to avoid flow leakage
        return [SlotSet("issue_description", None), SlotSet("order_id", None)]
class ActionLookupOrderStatusByEmail(Action):
    def name(self) -> Text:
        return "action_lookup_order_status_by_email"

    def run(self, dispatcher, tracker, domain):
        email = tracker.get_slot("lookup_email")
        # demo behavior: pretend the latest order for that email is ORD-12345
        if email:
            dispatcher.utter_message(text=f"Found your latest order via {email}: ORD-12345 — Shipped, ETA 2 days.")
        else:
            dispatcher.utter_message(text="I couldn’t find any orders for that email.")
        # clear temp slots
        return [SlotSet("lookup_email", None)]
class ActionUpdateAddress(Action):
    def name(self) -> Text: return "action_update_address"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        oid = tracker.get_slot("order_id"); addr = tracker.get_slot("new_address")
        if not oid or not addr:
            dispatcher.utter_message(text="I need both the order ID and the new address.")
            return []
        rec = ORDER_DB.get(oid)
        if not rec:
            dispatcher.utter_message(text=f"Order {oid} not found.")
            return clear_context_slots()
        if rec["status"].lower() == "shipped":
            dispatcher.utter_message(text=f"Order {oid} already shipped; I can’t change the address. Try carrier redirect.")
            return clear_context_slots()
        rec["address"] = addr
        dispatcher.utter_message(text=f"Address for {oid} updated to: {addr}.")
        return clear_context_slots()
# ---- Validators for new forms ----
class ValidateReturnForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_return_form"

    def validate_order_id(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """
        Validates the order ID during the return form.
        Handles: valid IDs, reuse of previous ID, and "I don't have it".
        """
        if not value:
            # if user didn’t provide a value, try to reuse previous
            previous_id = tracker.get_slot("order_id")
            if previous_id:
                dispatcher.utter_message(text=f"Got it — we’ll use your previous order ID ({previous_id}).")
                return {"order_id": previous_id}
            else:
                dispatcher.utter_message(text="No worries, kindly provide your order ID (e.g., ORD-12345).")
                return {"order_id": None, "lookup_email": None}

        text = value.strip().lower()

        # Handle the case where user says "I don't have it"
        if "don't have" in text or "do not have" in text:
            dispatcher.utter_message(text="No worries — let’s find your order by email instead.")
            return {"order_id": None, "lookup_email": None}

        # Check if valid format (e.g., ORD-12345)
        if re.match(r"^ord-\d{5}$", text, re.IGNORECASE):
            order_id = text.upper()
            return {"order_id": order_id}

        # Invalid format fallback
        dispatcher.utter_message(text="That doesn’t look like a valid order ID (e.g., ORD-12345). Could you recheck it?")
        return {"order_id": None}

class ValidateCancelOrderForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_cancel_order_form"

    def validate_order_id(self, value, dispatcher, tracker, domain) -> Dict[Text, Any]:
        out = _infer_emotion_slots(tracker)  # ← add emotion
        if tracker.get_slot("requested_slot") != "order_id":
            out["order_id"] = tracker.get_slot("order_id")
            return out

        if not value:
            out["order_id"] = None
            return out

        v = (value or "").strip().upper().replace(" ", "")
        if re.fullmatch(r"ORD-\d{5}", v):
            out["order_id"] = v
            return out

        dispatcher.utter_message(response="utter_order_id_invalid")
        out["order_id"] = None
        return out

class ValidateOrderLookupByEmailForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_order_lookup_by_email_form"

    def validate_lookup_email(self, value: Text, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict):
        if value and EMAIL_RE.match(value.strip()):
            return {"lookup_email": value.strip()}
        dispatcher.utter_message(response="utter_email_invalid")
        return {"lookup_email": None}
class ActionAfterReuseConfirm(Action):
    def name(self) -> Text:
        return "action_after_reuse_confirm"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict):
        reuse = tracker.get_slot("reuse_ok")
        events = [SlotSet("reuse_ok", None)]  # clear temp slot
        if reuse == "no":
            events.append(SlotSet("order_id", None))  # force asking new order id
            dispatcher.utter_message(text="No problem—let’s use a different order.")
        else:
            dispatcher.utter_message(text="Great, I’ll use the same order.")
        return events
def _latest_intent_name(tracker: Tracker) -> str | None:
    latest = tracker.latest_message or {}
    intent = latest.get("intent") or {}
    return intent.get("name")
def _reuse_ok_common(value: Text, tracker: Tracker, dispatcher: CollectingDispatcher) -> Dict[Text, Any]:
    v = (value or "").strip().lower()
    if v in {"yes", "y"}: return {"reuse_ok": "yes"}
    if v in {"no", "n"}:  return {"reuse_ok": "no"}
    i = _latest_intent_name(tracker)
    if i == "affirm": return {"reuse_ok": "yes"}
    if i == "deny":   return {"reuse_ok": "no"}
    dispatcher.utter_message(text="Please tap Yes or No.")
    return {"reuse_ok": None}

class ValidateConfirmReuseAddressForm(FormValidationAction):
    def name(self) -> str:
        return "validate_confirm_reuse_address_form"

    def validate_reuse_ok(self, value, dispatcher, tracker, domain):
        return _reuse_ok_common(value, tracker, dispatcher)


class ValidateConfirmReuseDamageForm(FormValidationAction):
    def name(self) -> str:
        return "validate_confirm_reuse_damage_form"

    def validate_reuse_ok(self, value, dispatcher, tracker, domain):
        return _reuse_ok_common(value, tracker, dispatcher)
class ValidateAddressUpdateForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_address_update_form"

    def validate_order_id(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        out = _infer_emotion_slots(tracker)  # ← add emotion
        requested = tracker.get_slot("requested_slot")
        current = tracker.get_slot("order_id")

        if requested != "order_id":
            out["order_id"] = current
            return out

        if not value:
            out["order_id"] = None
            return out

        v = normalize_order_id(value)
        if v and ORDER_RE.fullmatch(v):
            out["order_id"] = v
            return out

        dispatcher.utter_message(response="utter_order_id_invalid")
        out["order_id"] = None
        return out

    def validate_new_address(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        out = _infer_emotion_slots(tracker)  # ← add emotion
        requested = tracker.get_slot("requested_slot")

        if requested != "new_address":
            out["new_address"] = tracker.get_slot("new_address")
            return out

        if not value or len(value.strip()) < 8:
            dispatcher.utter_message(text="Please provide the full address including city & ZIP.")
            out["new_address"] = None
            return out

        out["new_address"] = value.strip()
        return out
