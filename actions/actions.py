from rasa_sdk.events import SessionStarted, ActionExecuted, EventType
from typing import Any, Text, Dict, List, Optional
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
import re
import os
import requests
import csv
from rasa_sdk.events import SlotSet, EventType, FollowupAction
import logging
from threading import Lock
from components.emotion import ensure_emotion_pipe, coarse_from_label, top_emotion 
from components.criticality import infer_criticality_for_text
from datetime import datetime, timedelta
from pathlib import Path
from components.absa import extract_aspect_opinions
NOW = datetime.now()
_init_lock = Lock()
DASHBOARD_BACKEND_URL = "http://localhost:5001"  # adjust if needed
# Fake database for demo
ORDER_DB = {
    # Classic shipped order – cannot cancel, can start return later
    "ORD-12345": {
        "status": "Shipped",
        "eta": "2 days",
        "created_at": (NOW - timedelta(days=2)).isoformat(),
        "address": "123 Main St, Nicosia, 10001",
        "email": "alex@example.com",
        "total": 89.90,
    },
    # Recently placed, still processing – can cancel
    "ORD-98765": {
        "status": "Processing",
        "eta": "4 days",
        "created_at": (NOW - timedelta(hours=3)).isoformat(),
        "address": "45 Green Rd, Larnaca, 6023",
        "email": "maria@example.com",
        "total": 149.50,
    },
    # Delivered order – for return / damage scenarios
    "ORD-55555": {
        "status": "Delivered",
        "eta": "Delivered",
        "created_at": (NOW - timedelta(days=5)).isoformat(),
        "address": "10 Beach Ave, Limassol, 3300",
        "email": "john@example.com",
        "total": 59.00,
    },
    # On hold – useful for “update address” scenarios
    "ORD-77777": {
        "status": "On hold",
        "eta": "TBD",
        "created_at": (NOW - timedelta(hours=10)).isoformat(),
        "address": "78 Lake St, Paphos, 8100",
        "email": "sofia@example.com",
        "total": 220.00,
    },
}

ORDER_RE = re.compile(r"^ORD-\d{5}$", re.IGNORECASE)
EMAIL_RE = re.compile(r"[^@]+@[^@]+\.[^@]+")

logger = logging.getLogger(__name__)

_CRIT2SCORE = {"low": 0.3, "medium": 0.5, "high": 0.75}
# Conversation-level thresholds are intentionally higher than per-turn thresholds.
# A single medium turn (turn-level score ~0.70) must NOT push the conversation
# to HIGH — that requires sustained distress across multiple turns.
_SCORE2CRIT = lambda s: ("high" if s>=0.75 else "medium" if s>=0.50 else "low")
_ESCALATION_MAP = {"low":"none","medium":"potential","high":"escalate_now"}

# ── Phase 2: in-process ABSA accumulator ────────────────────────────────────
# _infer_absa_for_latest_user_turn cannot SlotSet (it has no return path), so
# it writes here; ActionPushStateToDashboard reads and forwards to the backend.
_absa_cache: Dict[str, list] = {}


def _build_absa_insights(rows: list) -> Dict[str, Any]:
    """
    Roll up raw aspect rows into a manager-readable insights dict.

    Output structure (per business aspect):
      {
        "Customer Support": {
          "business_aspect": "customer_support",
          "display":         "Customer Support",
          "mentions":        5,
          "positive":        1,
          "negative":        3,
          "neutral":         1,
          "dominant_sentiment": "negative",
          "top_opinion_terms":  ["rude", "unhelpful", "not helpful"],
          "mean_confidence":    0.84,
          "emotions":           {"frustrated": 2, "disappointed": 1},
        },
        ...
      }

    Also includes a top-level "summary" key with the 3 most-mentioned
    negative business aspects for the dashboard alert strip.
    """
    from collections import defaultdict, Counter

    buckets: Dict[str, Dict] = defaultdict(lambda: {
        "positive": 0, "negative": 0, "neutral": 0,
        "opinion_terms": [], "confidences": [], "emotions": Counter(),
        "business_aspect": "", "display": "",
    })

    for row in rows:
        biz = row.get("business") or "overall_experience"
        display = row.get("business_display") or biz.replace("_", " ").title()
        sent = row.get("sentiment") or "neutral"
        conf = float(row.get("confidence") or 0.5)
        opinion = row.get("opinion_term")
        emotion = row.get("emotion")

        b = buckets[biz]
        b["business_aspect"] = biz
        b["display"] = display
        b[sent] = b.get(sent, 0) + 1
        if opinion:
            b["opinion_terms"].append(opinion)
        b["confidences"].append(conf)
        if emotion:
            b["emotions"][emotion] += 1

    result = {}
    for biz, b in buckets.items():
        pos = b["positive"]
        neg = b["negative"]
        neu = b["neutral"]
        total = pos + neg + neu

        opinions = b["opinion_terms"]
        # Deduplicate, keep top-3 most frequent
        top_opinions = [t for t, _ in Counter(opinions).most_common(3)]

        confs = b["confidences"]
        mean_conf = round(sum(confs) / len(confs), 3) if confs else 0.0

        dominant = "neutral"
        if neg > pos and neg >= neu:
            dominant = "negative"
        elif pos > neg and pos >= neu:
            dominant = "positive"

        top_emotion = b["emotions"].most_common(1)[0][0] if b["emotions"] else None

        result[b["display"]] = {
            "business_aspect":    biz,
            "display":            b["display"],
            "mentions":           total,
            "positive":           pos,
            "negative":           neg,
            "neutral":            neu,
            "dominant_sentiment": dominant,
            "top_opinion_terms":  top_opinions,
            "mean_confidence":    mean_conf,
            "top_emotion":        top_emotion,
        }

    # Top-3 negative issues for alert strip (sorted by negative count desc)
    negative_issues = sorted(
        [(k, v) for k, v in result.items() if v["dominant_sentiment"] == "negative"],
        key=lambda x: x[1]["negative"],
        reverse=True,
    )[:3]
    result["__summary__"] = {
        "total_aspects_detected": sum(v["mentions"] for k, v in result.items() if not k.startswith("__")),
        "top_negative_issues":    [k for k, _ in negative_issues],
    }

    return result
ABSA_ENABLED_INTENTS = {
    "product_feedback_positive",
    "product_feedback_negative",
    "report_damaged_product",
    "start_return",
    "cancel_order",
    "refund_policy",
    "shipping_times",
}
def _final_sentiment_from_conv_crit(conv_level: Optional[str]) -> str:
    level = (conv_level or "medium").lower()
    if level == "low":
        return "positive"
    if level == "high":
        return "negative"
    return "neutral"
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
    """Infer emotion + criticality for the latest user message and update conv-risk.

    Used by validators so that every turn inside forms still updates:
      - top_emotion / sentiment / overall_emotion
      - criticality_level / criticality_score / escalation
      - conversation_criticality / conversation_criticality_score / conversation_escalation

    ABSA (aspect-based sentiment) is run for every user turn, but rows are only
    exported if the message contains known aspect terms.
    """
    text = (tracker.latest_message.get("text") or "").strip()
    if not text:
        return {}

    intent = tracker.latest_message.get("intent", {}).get("name")

    # 1) Emotion
    pipe = _ensure_pipe()
    preds = pipe(text)[0]
    top = max(preds, key=lambda x: x["score"])
    top_label = top["label"]
    sentiment = _coarse(top_label)

    agg = _merge_emotion_aggregate(tracker, top_label)
    overall = agg.get("overall_emotion")

    # 2) Criticality (use emotion as prior for the model)
    crit = _infer_criticality_slots(text, top_label)
    level = crit["criticality_level"]
    score = float(crit["criticality_score"])
    escalation = crit["escalation"]

    # 3) Conversation-level risk update (uses ONLY criticality)
    conv_state = _update_conversation_risk(
        tracker, level, score,
        reason_code=crit.get("criticality_reason_code", ""),
        explanation=crit.get("criticality_explanation", ""),
    )

    # logger.info(
    #     "[form-emotion+crit] text=%r intent=%s -> emo=%s sentiment=%s overall=%s | "
    #     "crit_level=%s crit_score=%.2f crit_escalation=%s | conv=%s",
    #     text,
    #     intent,
    #     top_label,
    #     sentiment,
    #     overall,
    #     level,
    #     score,
    #     escalation,
    #     {
    #         "conversation_criticality": conv_state["conversation_criticality"],
    #         "conversation_criticality_score": conv_state["conversation_criticality_score"],
    #         "conversation_escalation": conv_state["conversation_escalation"],
    #     },
    # )

    # 4) ABSA: always run; extractor will decide whether to export
    try:
        _infer_absa_for_latest_user_turn(
            tracker=tracker,
            text=text,
            emotion_label=top_label,
            sentiment_label=sentiment,
        )
    except Exception as e:
        logger.exception("ABSA analysis failed: %s", e)

    out = {
        # emotion (business-friendly bucket + smoothing reason)
        "top_emotion": top_label,
        "sentiment": sentiment,
        "emotion_tally": agg["emotion_tally"],
        "emotion_history": agg["emotion_history"],
        "overall_emotion": agg.get("overall_emotion"),
        "emotion_smoothing_reason": agg.get("emotion_smoothing_reason", ""),
        # per-turn criticality + reason codes
        "criticality_level": level,
        "criticality_score": score,
        "escalation": escalation,
        "criticality_reason_code": crit.get("criticality_reason_code", ""),
        "criticality_explanation": crit.get("criticality_explanation", ""),
        # conversation-level risk (audit fields propagated below)
    }

    # add conversation-level risk slots (includes audit_log + false_trigger)
    out.update(conv_state)
    return out

def _infer_criticality_slots(text: str, emotion: Optional[str]) -> Dict[Text, Any]:
    text = (text or "").strip()
    if not text:
        # safe default
        return {
            "criticality_level": "low",
            "criticality_score": _CRIT2SCORE["low"],
            "escalation": _ESCALATION_MAP["low"],
        }

    result = infer_criticality_for_text(
        text,
        emotion,
        boost_from_emotion=True,   # or False if you decide to ignore emotion prior
        heuristics_enabled=True,   # <-- this enables YAML keyword overrides
    )

    level = (result.get("criticality_level") or "low").lower()
    score = float(result.get("criticality_score") or _CRIT2SCORE.get(level, 0.2))
    escalation = result.get("escalation") or _ESCALATION_MAP.get(level, "none")
    dist = result.get("distribution") or {}
    source = result.get("source") or "unknown"
    kw = result.get("matched_keyword")

    # logger.debug(
    #     "[crit] text=%r emo=%s -> level=%s score=%.2f escalation=%s source=%s kw=%s dist=%s",
    #     text,
    #     (emotion or "none"),
    #     level,
    #     score,
    #     escalation,
    #     source,
    #     kw,
    #     {k: f"{v:.2f}" for k, v in dist.items()},
    # )

    return {
        "criticality_level": level,
        "criticality_score": score,
        "escalation": escalation,
        "criticality_reason_code": result.get("reason_code", ""),    # Task 4
        "criticality_explanation": result.get("explanation", ""),     # Task 4
    }
def _update_conversation_risk(
    tracker: Tracker,
    level: str,
    score: float,
    reason_code: str = "",
    explanation: str = "",
) -> Dict[Text, Any]:
    """
    Update conversation-level risk based on this turn's criticality.

    Week 2 additions:
    ─────────────────
    Task 2 │ FALSE-TRIGGER PENALTY
           │ If the previous N turns were all HIGH but the conversation
           │ subsequently cooled (multiple LOW/MEDIUM turns in a row without
           │ further high signals), we apply a penalty that pushes the
           │ aggregated score DOWN, preventing "stuck escalation" and reducing
           │ alert fatigue.
           │
           │ Rule: if the last _FALSE_TRIGGER_GRACE consecutive turns are all
           │ ≤ medium AND the tally shows prior HIGH turns that never resulted
           │ in an agent action, we subtract _FALSE_TRIGGER_PENALTY from the
           │ aggregated score and record a "false_trigger_candidate" flag.
           │
    Task 3 │ CONFIDENCE DECAY WHEN CONVERSATION COOLS
           │ Base decay factor 0.85 (Week 1). Week 2 makes decay ADAPTIVE:
           │   – If last 3 turns are all LOW  → decay = 0.70 (fast cooling)
           │   – If last 3 turns are all HIGH → decay = 0.95 (slow cooling)
           │   – Otherwise                   → decay = 0.85 (baseline)
           │ This means a single early HIGH does not hold the conversation
           │ score elevated forever; it fades appropriately.
           │
    Task 4 │ ESCALATION AUDIT TRAIL
           │ Every turn appends a structured event to `escalation_audit_log`
           │ (a slot that persists for the session). Each event records:
           │   timestamp, turn, level, score, escalation, reason_code,
           │   explanation, source, matched_keyword, emotion, conv_score.
           │ The audit log is the backbone of the Escalation Audit Trail output.
    """
    import json
    from datetime import datetime as _dt

    # ── 0) LOAD HISTORY ─────────────────────────────────────────────────────
    history: List[Dict[str, Any]] = list(tracker.get_slot("criticality_history") or [])
    tally:   Dict[str, int]       = dict(tracker.get_slot("criticality_tally") or {})
    audit_log: List[Dict[str, Any]] = list(tracker.get_slot("escalation_audit_log") or [])

    turn_number = len(history) + 1

    history.append({"level": level, "score": float(score)})
    history = history[-15:]  # keep last 15 turns
    tally[level] = tally.get(level, 0) + 1

    # ── 1) ADAPTIVE DECAY (Task 3) ───────────────────────────────────────────
    recent3 = [h.get("level") for h in history[-3:]]
    if all(l == "low" for l in recent3):
        decay = 0.70   # conversation clearly cooled → fast decay
    elif all(l == "high" for l in recent3):
        decay = 0.95   # sustained distress → slow decay (don't dismiss too fast)
    else:
        decay = 0.85   # baseline

    w      = 1.0
    total_w = 0.0
    agg    = 0.0
    for item in reversed(history):
        s = float(item.get("score", 0.0))
        agg    += w * s
        total_w += w
        w      *= decay

    conv_score = agg / max(total_w, 1e-9)

    # ── 2) FALSE-TRIGGER PENALTY (Task 2) ────────────────────────────────────
    # Grace period: how many consecutive non-HIGH turns before we penalise a
    # prior high that was never confirmed by an agent action.
    _FALSE_TRIGGER_GRACE   = 4    # consecutive LOW/MEDIUM turns
    _FALSE_TRIGGER_PENALTY = 0.12  # flat subtraction from conv_score

    false_trigger_candidate = False
    recent_levels = [h.get("level") for h in history[-_FALSE_TRIGGER_GRACE:]]
    prior_high = tally.get("high", 0)
    agent_acted = bool(tracker.get_slot("handoff_requested"))

    if (
        len(recent_levels) == _FALSE_TRIGGER_GRACE
        and all(l in ("low", "medium") for l in recent_levels)
        and prior_high > 0
        and not agent_acted
    ):
        conv_score = max(0.0, conv_score - _FALSE_TRIGGER_PENALTY)
        false_trigger_candidate = True
        logger.info(
            "[conv-risk] FALSE_TRIGGER_PENALTY applied: "
            "prior_high=%d grace_turns=%s new_conv_score=%.2f session=%s",
            prior_high, recent_levels, conv_score, tracker.sender_id,
        )

    # ── 3) MAP SCORE → LEVEL ─────────────────────────────────────────────────
    conv_level     = _SCORE2CRIT(conv_score)
    conv_escalation = _ESCALATION_MAP[conv_level]

    # Guardrail: only lock in escalate_now if there was a genuine per-turn HIGH
    # (not just conv_score drift from accumulated medium turns).
    # false_trigger_candidate cancels this lock.
    prior_turn_highs = [h for h in history if h.get("level") == "high"]
    if prior_turn_highs and not false_trigger_candidate:
        conv_escalation = "escalate_now"
        conv_level      = "high"
        conv_score      = max(conv_score, 0.9)

    # ── 4) ESCALATION AUDIT LOG (Task 4) ─────────────────────────────────────
    # Every turn is logged. Dashboard / ops team can filter on escalation != "none".
    emo   = tracker.get_slot("top_emotion") or "unknown"
    text  = (tracker.latest_message.get("text") or "").strip()

    audit_event: Dict[str, Any] = {
        "ts":              _dt.utcnow().isoformat(timespec="seconds") + "Z",
        "session_id":      tracker.sender_id,
        "turn":            turn_number,
        # Turn-level signal
        "turn_level":      level,
        "turn_score":      round(float(score), 3),
        "turn_escalation": _ESCALATION_MAP.get(level, "none"),
        # Conversation-level aggregated signal
        "conv_level":      conv_level,
        "conv_score":      round(conv_score, 3),
        "conv_escalation": conv_escalation,
        # Explainability (from Week 1)
        "reason_code":     reason_code,
        "explanation":     explanation,
        # Context
        "emotion":         emo,
        "text_preview":    text[:120],
        # Week 2 metadata
        "decay_used":      decay,
        "false_trigger_candidate": false_trigger_candidate,
        "history_snapshot": [h.get("level") for h in history[-5:]],
    }
    audit_log.append(audit_event)
    audit_log = audit_log[-50:]  # keep last 50 events per session

    # Structured log line — one grep-able line per escalation event
    if conv_escalation != "none" or level == "high":
        logger.info(
            "[ESCALATION_AUDIT] session=%s turn=%d "
            "turn_level=%s turn_score=%.2f "
            "conv_level=%s conv_score=%.2f conv_esc=%s "
            "reason=%s decay=%.2f false_trigger=%s emotion=%s text=%r",
            tracker.sender_id,
            turn_number,
            level, float(score),
            conv_level, conv_score, conv_escalation,
            reason_code,
            decay,
            false_trigger_candidate,
            emo,
            text[:80],
        )
    else:
        logger.debug(
            "[conv-risk] session=%s turn=%d level=%s score=%.2f conv=%s/%.2f decay=%.2f",
            tracker.sender_id, turn_number, level, float(score),
            conv_level, conv_score, decay,
        )

    return {
        "criticality_history":     history,
        "criticality_tally":       tally,
        "conversation_criticality": conv_level,
        "conversation_criticality_score": conv_score,
        "conversation_escalation": conv_escalation,
        "escalation_audit_log":    audit_log,            # Task 4
        "false_trigger_candidate": false_trigger_candidate,  # Task 2
    }
def _best_overall_from_tally(tally: Dict[str, int], last_top: str) -> str:
    if not tally:
        return last_top or "neutral"
    # max count; break ties in favor of the most recent top label
    maxc = max(tally.values())
    candidates = [e for e, c in tally.items() if c == maxc]
    return last_top if last_top in candidates else sorted(candidates)[0]

def _merge_emotion_aggregate(tracker, new_label):
    """
    Week 1 Task 2: Rolling emotion smoothing per session.

    Applies a recency-weighted window (last 4 turns) so a single ambiguous
    message cannot flip the displayed emotion label. Uses the smooth_emotion()
    function from emotion.py.
    """
    from components.emotion import smooth_emotion, BUSINESS_EMOTION_DISPLAY

    tally   = tracker.get_slot("emotion_tally") or {}
    history = list(tracker.get_slot("emotion_history") or [])

    tally[new_label] = tally.get(new_label, 0) + 1

    # Apply smoothing BEFORE appending so history = prior turns
    smoothed_label, smooth_reason = smooth_emotion(new_label, history)

    history.append(smoothed_label)   # store smoothed value in history
    if len(history) > 10:
        history.pop(0)

    # overall_emotion = smoothed label for this turn
    overall = smoothed_label

    logger.debug(
        "[emotion-smooth] raw=%s smoothed=%s reason=%s history=%s",
        new_label,
        smoothed_label,
        smooth_reason,
        history[-4:],
    )

    return {
        "emotion_tally": tally,
        "emotion_history": history,
        "overall_emotion": overall,
        "emotion_smoothing_reason": smooth_reason,   # reason code for transparency
    }
#ABSA #############HELPER FUNCTIONS#####################
def _build_absa_rows_for_message(
    tracker: Tracker,
    text: str,
    aspects: List[Dict[str, Any]],
    emotion_label: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Convert aspect-opinion extraction results + current message into CSV rows.

    Dataset schema (Phase 2 additions marked with *):
      Session_ID, Original_Text, Aspect, Aspect_Term, Opinion_Term,
      Aspect_Sentiment, Confidence, Category,
      Business_Aspect*, Business_Aspect_Display*,   ← manager-level roll-up
      Match_Type*, Similarity*,                     ← explainability fields
      Emotion
    """
    original_text = (text or "").strip()
    if not original_text or not aspects:
        return []

    session_id = tracker.sender_id

    rows: List[Dict[str, Any]] = []
    for a in aspects:
        rows.append(
            {
                "Session_ID":             session_id,
                "Original_Text":          original_text,
                # Fine-grained aspect (unchanged)
                "Aspect":                 a.get("aspect"),
                "Aspect_Term":            a.get("aspect_term"),
                "Opinion_Term":           a.get("opinion_term"),
                "Aspect_Sentiment":       a.get("aspect_sentiment"),
                "Confidence":             a.get("aspect_sentiment_confidence"),
                "Category":               a.get("aspect_category"),
                # Phase 2: business-level roll-up
                "Business_Aspect":        a.get("business_aspect"),
                "Business_Aspect_Display": a.get("business_aspect_display"),
                # Phase 2: explainability
                "Match_Type":             a.get("aspect_match_type", "exact"),
                "Similarity":             a.get("aspect_similarity", 1.0),
                # Emotion context
                "Emotion":                emotion_label,
            }
        )

    return rows

def _export_absa_rows(rows: List[Dict[str, Any]]) -> None:
    """
    Append ABSA rows to data/aspect_sentiment_sessions.csv, deduplicating by
    (Session_ID, Original_Text, Aspect, Opinion_Term).
    """
    if not rows:
        logger.info("[ABSA] No rows to export, skipping CSV write")
        return

    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    csv_path = base_dir / "data" / "aspect_sentiment_sessions.csv"

    logger.info("[ABSA] Exporting %d rows to %s", len(rows), csv_path)

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    existing_rows: List[Dict[str, Any]] = []
    existing_keys = set()

    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_rows.append(r)
                key = (
                    r.get("Session_ID"),
                    r.get("Original_Text"),
                    r.get("Aspect"),
                    r.get("Opinion_Term"),
                )
                existing_keys.add(key)

    new_unique_rows: List[Dict[str, Any]] = []
    for r in rows:
        key = (
            r.get("Session_ID"),
            r.get("Original_Text"),
            r.get("Aspect"),
            r.get("Opinion_Term"),
        )
        if key not in existing_keys:
            existing_keys.add(key)
            new_unique_rows.append(r)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in existing_rows:
            writer.writerow(r)
        for r in new_unique_rows:
            writer.writerow(r)

    logger.info(
        "[ABSA] CSV write complete: %d existing + %d new rows",
        len(existing_rows),
        len(new_unique_rows),
    )
def _infer_absa_for_latest_user_turn(
    tracker: Tracker,
    text: str,
    emotion_label: Optional[str],
    sentiment_label: Optional[str],
) -> None:
    text = (text or "").strip()
    if not text:
        return

    aspects = extract_aspect_opinions(
        text=text,
        sentiment_label=sentiment_label,
        emotion_label=emotion_label,
    )
    if not aspects:
        logger.debug("[ABSA] No aspects found in text=%r, skipping export", text)
        return

    rows = _build_absa_rows_for_message(
        tracker=tracker,
        text=text,
        aspects=aspects,
        emotion_label=emotion_label,
    )
    _export_absa_rows(rows)

    # ── Phase 2: accumulate aspects into session slot for dashboard insights ──
    # We store lightweight dicts (no Original_Text to keep payload small).
    existing: list = list(tracker.get_slot("absa_session_aspects") or [])
    for a in aspects:
        existing.append({
            "aspect":          a.get("aspect"),
            "business":        a.get("business_aspect"),
            "business_display": a.get("business_aspect_display"),
            "sentiment":       a.get("aspect_sentiment"),
            "confidence":      a.get("aspect_sentiment_confidence"),
            "opinion_term":    a.get("opinion_term"),
            "match_type":      a.get("aspect_match_type", "exact"),
            "emotion":         emotion_label,
        })
    # cap at 200 rows per session to stay within slot memory
    if len(existing) > 200:
        existing = existing[-200:]
    # We cannot SlotSet here (no return path) — store on tracker via a module-level
    # cache keyed by sender_id; ActionPushStateToDashboard reads it.
    _absa_cache[tracker.sender_id] = existing
############################################
class ActionUpdateConversationRisk(Action):
    def name(self) -> str: 
        return "action_update_conversation_risk"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        # TURN-LEVEL CRITICALITY
        text = (tracker.latest_message.get("text") or "").strip()
        emo_for_prior = tracker.get_slot("top_emotion")

        crit = _infer_criticality_slots(text, emo_for_prior)
        level = (crit.get("criticality_level") or "low").lower()
        score = float(crit.get("criticality_score") or _CRIT2SCORE.get(level, 0.2))
        turn_escalation = crit.get("escalation") or _ESCALATION_MAP.get(level, "none")

        # CONVERSATION-LEVEL RISK (shared helper)
        conv_state = _update_conversation_risk(
            tracker, level, score,
            reason_code=crit.get("criticality_reason_code", ""),
            explanation=crit.get("criticality_explanation", ""),
        )

        events: List[EventType] = [
            SlotSet("criticality_level", level),
            SlotSet("criticality_score", score),
            SlotSet("escalation", turn_escalation),
            SlotSet("criticality_reason_code", crit.get("criticality_reason_code", "")),
            SlotSet("criticality_explanation", crit.get("criticality_explanation", "")),
        ]

        for k, v in conv_state.items():
            events.append(SlotSet(k, v))

        # ---- NEW: trigger escalation immediately if needed ----
        conv_escalation = conv_state.get("conversation_escalation") or "none"
        already_handed_off = bool(tracker.get_slot("handoff_requested"))

        if conv_escalation == "escalate_now" and not already_handed_off:
            # Schedule the escalation action, which will utter_handoff_ack
            events.append(FollowupAction("action_escalate_now"))

        return events

class ActionEscalateNow(Action):
    def name(self) -> Text:
        return "action_escalate_now"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(response="utter_handoff_ack")

        events: List[EventType] = [
            SlotSet("resolution_status", "escalated"),
            SlotSet("handoff_requested", True),
            FollowupAction("action_push_state_to_dashboard"),  # <-- add this
        ]

        export_events = ActionExportConversationSummary().run(dispatcher, tracker, domain)
        events.extend(export_events)

        return events
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
        return [SlotSet("resolution_status", "escalated")]
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
            SlotSet("emotion_history", agg["emotion_history"]),  # <-- add this
            SlotSet("overall_emotion", overall),
        ]


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
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """
        Validates the order ID during the return form.
        Handles: valid IDs, reuse of previous ID, and "I don't have it".
        Also updates emotion + conv-risk for this turn.
        """
        out = _infer_emotion_slots(tracker)  # ← emotion + criticality + conv-risk

        # If this slot is not actively requested, just keep the current value
        if tracker.get_slot("requested_slot") != "order_id":
            out["order_id"] = tracker.get_slot("order_id")
            return out

        if not value:
            previous_id = tracker.get_slot("order_id")
            if previous_id:
                dispatcher.utter_message(text=f"Got it — we’ll use your previous order ID ({previous_id}).")
                out["order_id"] = previous_id
                return out
            else:
                dispatcher.utter_message(text="No worries, kindly provide your order ID (e.g., ORD-12345).")
                out["order_id"] = None
                out["lookup_email"] = None
                return out

        text = value.strip().lower()

        # Valid ID format
        if re.match(r"^ord-\d{5}$", text, re.IGNORECASE):
            order_id = text.upper()
            out["order_id"] = order_id
            return out

        # Invalid format fallback
        dispatcher.utter_message(text="That doesn’t look like a valid order ID (e.g., ORD-12345). Could you recheck it?")
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
    def name(self) -> Text: 
        return "action_cancel_order"

    def run(self, dispatcher, tracker, domain):
        oid = tracker.get_slot("order_id")
        confirm = tracker.get_slot("confirm_cancel")

        # 1) Missing order ID → fallback + export
        if not oid:
            dispatcher.utter_message(text="I need the order ID to cancel.")
            events: List[EventType] = [SlotSet("resolution_status", "fallback")]
            export_events = ActionExportConversationSummary().run(
                dispatcher, tracker, domain
            )
            events.extend(export_events)
            return events

        # 2) User does NOT confirm → resolved + export
        if not confirm:
            dispatcher.utter_message(text="Okay, I’ll keep the order as is.")
            events: List[EventType] = clear_context_slots() + [
                SlotSet("resolution_status", "resolved")
            ]
            export_events = ActionExportConversationSummary().run(
                dispatcher, tracker, domain
            )
            events.extend(export_events)
            return events

        rec = ORDER_DB.get(oid)

        # 3) Order not found → fallback + export
        if not rec:
            dispatcher.utter_message(text=f"Order {oid} not found.")
            events: List[EventType] = clear_context_slots() + [
                SlotSet("resolution_status", "fallback")
            ]
            export_events = ActionExportConversationSummary().run(
                dispatcher, tracker, domain
            )
            events.extend(export_events)
            return events

        created = datetime.fromisoformat(rec.get("created_at", "2025-10-01"))

        # 4) Cannot cancel (too late / shipped) → fallback + export
        if rec["status"].lower() == "shipped" or datetime.now() - created > timedelta(hours=24):
            dispatcher.utter_message(
                text=f"Order {oid} can’t be canceled now. You can start a return after delivery."
            )
            events: List[EventType] = clear_context_slots() + [
                SlotSet("resolution_status", "fallback")
            ]
            export_events = ActionExportConversationSummary().run(
                dispatcher, tracker, domain
            )
            events.extend(export_events)
            return events

        # 5) Successfully canceled → resolved + export
        dispatcher.utter_message(
            text=f"Order {oid} has been canceled and refunded. You’ll get your refund in 3–5 working days."
        )
        events: List[EventType] = clear_context_slots() + [
            SlotSet("resolution_status", "resolved")
        ]
        export_events = ActionExportConversationSummary().run(
            dispatcher, tracker, domain
        )
        events.extend(export_events)
        return events


class ActionStartReturn(Action):
    def name(self) -> Text:
        return "action_start_return"

    def run(self, dispatcher, tracker, domain) -> List[EventType]:
        events: List[EventType] = []

        # 0) Infer emotion + criticality + conv-risk for this turn
        # (same helper you use in ValidateReturnForm)
        emotion_slots = _infer_emotion_slots(tracker)
        for slot_name, slot_value in emotion_slots.items():
            events.append(SlotSet(slot_name, slot_value))

        oid = tracker.get_slot("order_id")
        reason = tracker.get_slot("return_reason")

        # 1) Missing details → fallback + export
        if not oid or not reason:
            dispatcher.utter_message(text="I’m missing details for the return.")
            events.append(SlotSet("resolution_status", "fallback"))

            export_events = ActionExportConversationSummary().run(
                dispatcher, tracker, domain
            )
            events.extend(export_events)
            return events

        # 2) Order not found → fallback + export
        if oid not in ORDER_DB:
            dispatcher.utter_message(text=f"Order {oid} not found.")
            events.extend(clear_context_slots())
            events.append(SlotSet("resolution_status", "fallback"))

            export_events = ActionExportConversationSummary().run(
                dispatcher, tracker, domain
            )
            events.extend(export_events)
            return events

        # 3) Successful return → resolved + export
        dispatcher.utter_message(
            text=(
                f"Return initiated for {oid} "
                f"(reason: {reason}). You’ll receive a label by email shortly."
            )
        )
        events.extend(clear_context_slots())
        events.append(SlotSet("resolution_status", "resolved"))

        export_events = ActionExportConversationSummary().run(
            dispatcher, tracker, domain
        )
        events.extend(export_events)

        return events
# --- Actions ---
class ActionLookupOrderStatusByEmail(Action):
    def name(self) -> Text:
        return "action_lookup_order_status_by_email"

    def run(self, dispatcher, tracker, domain):
        email = (tracker.get_slot("lookup_email") or "").strip().lower()

        # Helper: find latest order for this email from ORDER_DB
        def _find_order_by_email(target_email: str):
            matches = []
            for oid, data in ORDER_DB.items():
                db_email = (data.get("email") or "").strip().lower()
                if db_email == target_email:
                    matches.append((oid, data))

            # If multiple matches exist, pick the newest by created_at (ISO string)
            if not matches:
                return None, None

            matches.sort(key=lambda x: x[1].get("created_at", ""), reverse=True)
            return matches[0]  # (order_id, data)

        events: List[EventType] = []

        if email:
            order_id, order = _find_order_by_email(email)

            if order_id and order:
                status = order.get("status", "Unknown")
                eta = order.get("eta", "TBD")

                dispatcher.utter_message(
                    text=f"Found your latest order via {email}: {order_id} — {status}, ETA {eta}."
                )
                events += [
                    SlotSet("lookup_email", None),
                    SlotSet("order_id", order_id),  # optional but useful for continuity
                    SlotSet("resolution_status", "resolved"),
                ]
            else:
                dispatcher.utter_message(text=f"I couldn’t find any orders for {email}.")
                events += [
                    SlotSet("lookup_email", None),
                    SlotSet("resolution_status", "fallback"),
                ]

            # ✅ Always ask for conversation feedback at the end of the email flow
            dispatcher.utter_message(response="utter_ask_conversation_feedback")

            # Export after the final bot message (so the log reflects the full outcome)
            export_events = ActionExportConversationSummary().run(dispatcher, tracker, domain)
            events.extend(export_events)
            return events

        # If somehow email is empty, keep previous behavior but also request feedback
        dispatcher.utter_message(text="Please share the email you used for your order.")
        dispatcher.utter_message(response="utter_ask_conversation_feedback")

        events += [
            SlotSet("lookup_email", None),
            SlotSet("resolution_status", "fallback"),
        ]
        export_events = ActionExportConversationSummary().run(dispatcher, tracker, domain)
        events.extend(export_events)
        return events

class ActionPasswordReset(Action):
    def name(self) -> Text:
        return "action_password_reset"

    def run(
        self, 
        dispatcher: CollectingDispatcher, 
        tracker: Tracker, 
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        email = tracker.get_slot("email")

        # Missing email → fallback + export
        if not email:
            dispatcher.utter_message(text="I need the account email to send a reset.")
            events: List[EventType] = [SlotSet("resolution_status", "fallback")]
            export_events = ActionExportConversationSummary().run(
                dispatcher, tracker, domain
            )
            events.extend(export_events)
            return events

        # Reset email sent → resolved + export
        dispatcher.utter_message(text=f"Got it. A password reset email has been sent to {email}.")
        events: List[EventType] = [
            SlotSet("issue_description", None),
            SlotSet("order_id", None),
            SlotSet("resolution_status", "resolved"),
        ]
        export_events = ActionExportConversationSummary().run(
            dispatcher, tracker, domain
        )
        events.extend(export_events)
        return events

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
        return [
            SlotSet("resolution_status", "escalated"),
            SlotSet("handoff_requested", True),
        ]

# Optional: clear context when user starts a new flow
class ActionClearContext(Action):
    def name(self) -> Text:
        return "action_clear_context"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        return clear_context_slots() + [SlotSet("emotion_tally", None), SlotSet("overall_emotion", None)]
class ActionLookupOrderStatus(Action):
    def name(self) -> Text:
        return "action_lookup_order_status"

    def run(
        self, 
        dispatcher: CollectingDispatcher, 
        tracker: Tracker, 
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        order_id = tracker.get_slot("order_id")

        # No order id → fallback + export
        if not order_id:
            dispatcher.utter_message(text="I couldn’t find an order ID. Please try again.")
            events: List[EventType] = [SlotSet("resolution_status", "fallback")]
            export_events = ActionExportConversationSummary().run(
                dispatcher, tracker, domain
            )
            events.extend(export_events)
            return events

        rec = ORDER_DB.get(order_id)

        # Not found → fallback + export
        if not rec:
            dispatcher.utter_message(
                text=f"Sorry, I can’t find {order_id}. Double-check the ID (e.g., ORD-12345)."
            )
            events = [SlotSet("resolution_status", "fallback")]
            export_events = ActionExportConversationSummary().run(
                dispatcher, tracker, domain
            )
            events.extend(export_events)
            return events

        # Found → resolved + export
        dispatcher.utter_message(
            text=f"Order {order_id}: {rec['status']} — ETA {rec['eta']}."
        )
        events = [SlotSet("resolution_status", "resolved")]
        export_events = ActionExportConversationSummary().run(
            dispatcher, tracker, domain
        )
        events.extend(export_events)
        return events

# --- Validators ---
class ValidateOrderStatusForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_order_status_form"

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
        Also updates emotion + conv-risk for this turn.
        """
        out = _infer_emotion_slots(tracker)  # ← emotion + criticality + conv-risk

        # If this slot is not actively requested, just keep the current value
        if tracker.get_slot("requested_slot") != "order_id":
            out["order_id"] = tracker.get_slot("order_id")
            return out

        if not value:
            previous_id = tracker.get_slot("order_id")
            if previous_id:
                dispatcher.utter_message(text=f"Got it — we’ll use your previous order ID ({previous_id}).")
                out["order_id"] = previous_id
                return out
            else:
                dispatcher.utter_message(text="No worries, kindly provide your order ID (e.g., ORD-12345).")
                out["order_id"] = None
                out["lookup_email"] = None
                return out

        text = value.strip().lower()
        # Valid ID format
        if re.match(r"^ord-\d{5}$", text, re.IGNORECASE):
            order_id = text.upper()
            out["order_id"] = order_id
            return out

        # Invalid format fallback
        dispatcher.utter_message(text="That doesn’t look like a valid order ID (e.g., ORD-12345). Could you recheck it?")
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

    def run(self, dispatcher, tracker, domain):
        order_id = tracker.get_slot("order_id")
        desc = tracker.get_slot("issue_description")

        # Missing details → fallback + export
        if not order_id or not desc:
            dispatcher.utter_message(
                text="I’m missing some details. Please provide the order ID and a short description."
            )
            events: List[EventType] = [SlotSet("resolution_status", "fallback")]
            export_events = ActionExportConversationSummary().run(
                dispatcher, tracker, domain
            )
            events.extend(export_events)
            return events

        # Ticket created → resolved + export
        dispatcher.utter_message(
            text=f"Ticket created for {order_id}: '{desc}'. A support agent will follow up shortly."
        )
        events: List[EventType] = [
            SlotSet("issue_description", None),
            SlotSet("order_id", None),
            SlotSet("resolution_status", "escalated"),
        ]
        export_events = ActionExportConversationSummary().run(
            dispatcher, tracker, domain
        )
        events.extend(export_events)
        return events


class ActionSessionStart(Action):
    def name(self) -> Text:
        return "action_session_start"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[EventType]:
        """Initialise analytics and conv-risk slots for each new session."""

        now = datetime.utcnow().isoformat()

        events: List[EventType] = [
            SessionStarted(),
            SlotSet("session_start_time", now),

            # reset export guard
            SlotSet("export_done", False),

            # reset high-level outcome slots
            SlotSet("primary_intent", None),
            SlotSet("resolution_status", None),
            SlotSet("final_sentiment", None),
            SlotSet("handoff_requested", False),

            # reset conversation-level risk
            SlotSet("conversation_criticality", "low"),
            SlotSet("conversation_criticality_score", 0.0),
            SlotSet("conversation_escalation", "none"),
            SlotSet("criticality_history", []),
            SlotSet("criticality_tally", {}),

            # reset emotion aggregates
            SlotSet("emotion_tally", {}),
            SlotSet("emotion_history", []),
            SlotSet("overall_emotion", None),
            SlotSet("top_emotion", None),
            SlotSet("sentiment", None),
        ]

        # let Rasa listen for the first user message of the new session
        events.append(ActionExecuted("action_listen"))
        return events
class ActionSetPrimaryIntent(Action):
    def name(self) -> Text:
        return "action_set_primary_intent"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[EventType]:
        intent_name = tracker.latest_message.get("intent", {}).get("name")
        events: List[EventType] = [
            SlotSet("primary_intent", intent_name),
        ]

        # --- NEW: ABSA hook for all feedback intents ---
        if intent_name in ABSA_ENABLED_INTENTS:
            text = (tracker.latest_message.get("text") or "").strip()
            emo = tracker.get_slot("top_emotion")
            sentiment = tracker.get_slot("sentiment")  # your coarse sentiment slot

            _infer_absa_for_latest_user_turn(
                tracker=tracker,
                text=text,
                emotion_label=emo,
                sentiment_label=sentiment,
            )

        return events
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
        Also updates emotion + conv-risk for this turn.
        """
        out = _infer_emotion_slots(tracker)  # ← emotion + criticality + conv-risk

        # If this slot is not actively requested, just keep the current value
        if tracker.get_slot("requested_slot") != "order_id":
            out["order_id"] = tracker.get_slot("order_id")
            return out

        if not value:
            previous_id = tracker.get_slot("order_id")
            if previous_id:
                dispatcher.utter_message(text=f"Got it — we’ll use your previous order ID ({previous_id}).")
                out["order_id"] = previous_id
                return out
            else:
                dispatcher.utter_message(text="No worries, kindly provide your order ID (e.g., ORD-12345).")
                out["order_id"] = None
                out["lookup_email"] = None
                return out

        text = value.strip().lower()

        # "I don't have it" → fallback to email
        if "don't have" in text or "do not have" in text:
            dispatcher.utter_message(text="No worries — let’s find your order by email instead.")
            out["order_id"] = None
            out["lookup_email"] = None
            return out

        # Valid ID format
        if re.match(r"^ord-\d{5}$", text, re.IGNORECASE):
            order_id = text.upper()
            out["order_id"] = order_id
            return out

        # Invalid format fallback
        dispatcher.utter_message(text="That doesn’t look like a valid order ID (e.g., ORD-12345). Could you recheck it?")
        out["order_id"] = None
        return out


class ValidateCancelOrderForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_cancel_order_form"

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
        Also updates emotion + conv-risk for this turn.
        """
        out = _infer_emotion_slots(tracker)  # ← emotion + criticality + conv-risk

        # If this slot is not actively requested, just keep the current value
        if tracker.get_slot("requested_slot") != "order_id":
            out["order_id"] = tracker.get_slot("order_id")
            return out

        if not value:
            previous_id = tracker.get_slot("order_id")
            if previous_id:
                dispatcher.utter_message(text=f"Got it — we’ll use your previous order ID ({previous_id}).")
                out["order_id"] = previous_id
                return out
            else:
                dispatcher.utter_message(text="No worries, kindly provide your order ID (e.g., ORD-12345).")
                out["order_id"] = None
                out["lookup_email"] = None
                return out

        text = value.strip().lower()
        # Valid ID format
        if re.match(r"^ord-\d{5}$", text, re.IGNORECASE):
            order_id = text.upper()
            out["order_id"] = order_id
            return out

        # Invalid format fallback
        dispatcher.utter_message(text="That doesn’t look like a valid order ID (e.g., ORD-12345). Could you recheck it?")
        out["order_id"] = None
        return out

class ValidateOrderLookupByEmailForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_order_lookup_by_email_form"

    def validate_lookup_email(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:

        # Always update emotion + criticality + conv-risk
        out = _infer_emotion_slots(tracker)

        # If not actively asking for this slot → keep old value
        if tracker.get_slot("requested_slot") != "lookup_email":
            out["lookup_email"] = tracker.get_slot("lookup_email")
            return out

        # Validate email
        if value and EMAIL_RE.fullmatch(value.strip()):
            out["lookup_email"] = value.strip()
            return out

        dispatcher.utter_message(response="utter_email_invalid")
        out["lookup_email"] = None
        return out

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

    def validate_reuse_ok(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:

        # Always update emotion + criticality + conv-risk
        out = _infer_emotion_slots(tracker)

        # If not the active slot, just keep its current value
        if tracker.get_slot("requested_slot") != "reuse_ok":
            out["reuse_ok"] = tracker.get_slot("reuse_ok")
            return out

        # Reuse existing logic
        v = (value or "").strip().lower()
        if v in {"yes", "y"}:
            out["reuse_ok"] = "yes"
            return out
        if v in {"no", "n"}:
            out["reuse_ok"] = "no"
            return out

        # If user tapped Yes/No button intents
        intent = tracker.latest_message.get("intent", {}).get("name")
        if intent == "affirm":
            out["reuse_ok"] = "yes"
            return out
        if intent == "deny":
            out["reuse_ok"] = "no"
            return out

        dispatcher.utter_message(text="Please tap Yes or No.")
        out["reuse_ok"] = None
        return out



class ValidateConfirmReuseDamageForm(FormValidationAction):
    def name(self) -> str:
        return "validate_confirm_reuse_damage_form"

    def validate_reuse_ok(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:

        # Always update emotion + criticality + conv-risk
        out = _infer_emotion_slots(tracker)

        # If this slot is not being requested, keep current value
        if tracker.get_slot("requested_slot") != "reuse_ok":
            out["reuse_ok"] = tracker.get_slot("reuse_ok")
            return out

        v = (value or "").strip().lower()
        if v in {"yes", "y"}:
            out["reuse_ok"] = "yes"
            return out
        if v in {"no", "n"}:
            out["reuse_ok"] = "no"
            return out

        # Handle affirm/deny intents
        intent = tracker.latest_message.get("intent", {}).get("name")
        if intent == "affirm":
            out["reuse_ok"] = "yes"
            return out
        if intent == "deny":
            out["reuse_ok"] = "no"
            return out

        dispatcher.utter_message(text="Please tap Yes or No.")
        out["reuse_ok"] = None
        return out

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
        """
        Validates the order ID during the return form.
        Handles: valid IDs, reuse of previous ID, and "I don't have it".
        Also updates emotion + conv-risk for this turn.
        """
        out = _infer_emotion_slots(tracker)  # ← emotion + criticality + conv-risk

        # If this slot is not actively requested, just keep the current value
        if tracker.get_slot("requested_slot") != "order_id":
            out["order_id"] = tracker.get_slot("order_id")
            return out

        if not value:
            previous_id = tracker.get_slot("order_id")
            if previous_id:
                dispatcher.utter_message(text=f"Got it — we’ll use your previous order ID ({previous_id}).")
                out["order_id"] = previous_id
                return out
            else:
                dispatcher.utter_message(text="No worries, kindly provide your order ID (e.g., ORD-12345).")
                out["order_id"] = None
                out["lookup_email"] = None
                return out

        text = value.strip().lower()
        # Valid ID format
        if re.match(r"^ord-\d{5}$", text, re.IGNORECASE):
            order_id = text.upper()
            out["order_id"] = order_id
            return out

        # Invalid format fallback
        dispatcher.utter_message(text="That doesn’t look like a valid order ID (e.g., ORD-12345). Could you recheck it?")
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
class ActionExportConversationSummary(Action):
    def name(self) -> Text:
        return "action_export_conversation_summary"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[EventType]:

        session_id = tracker.sender_id
        start_time = tracker.get_slot("session_start_time")
        primary_intent = tracker.get_slot("primary_intent") or "unknown"
        resolution_status = tracker.get_slot("resolution_status") or "unknown"
        conv_level = tracker.get_slot("conversation_criticality")
        conversation_feedback = "not_provided"
        # feedback slot (may be None if user didn't answer yet)
        if tracker.get_slot("conversation_feedback"):
            conversation_feedback = tracker.get_slot("conversation_feedback")
        #conversation_feedback = tracker.get_slot("conversation_feedback") or "not_provided"

        # if final_sentiment not set yet, derive from conversation criticality
        final_sentiment = tracker.get_slot("final_sentiment") or _final_sentiment_from_conv_crit(conv_level)
        handoff = bool(tracker.get_slot("handoff_requested") or False)

        # --- compute total_messages & duration from events ---
        events = list(tracker.events)
        first_ts = None
        last_ts = None
        total_messages = 0

        for e in events:
            etype = e.get("event")
            ts = e.get("timestamp")
            if etype in {"user", "bot"}:
                total_messages += 1
                if ts is not None:
                    if first_ts is None or ts < first_ts:
                        first_ts = ts
                    if last_ts is None or ts > last_ts:
                        last_ts = ts

        if first_ts is not None and last_ts is not None:
            duration_seconds = last_ts - first_ts
        else:
            duration_seconds = 0.0

        # if start_time slot is missing, reconstruct from first_ts
        if not start_time and first_ts is not None:
            start_time = datetime.utcfromtimestamp(first_ts).isoformat()

        row = {
            "Session_ID": session_id,
            "Start_time": start_time,
            "User_intent": primary_intent,
            "Resolution_Status": resolution_status,
            "Final_Sentiment": final_sentiment,
            "Handoff_to_agent": "yes" if handoff else "no",
            "total_messages": total_messages,
            "conversation_duration": round(duration_seconds),
            "Conversation_Feedback": conversation_feedback,
        }

        logger.info("[dataset] row=%s", row)

        csv_path = os.path.join("data", "conversation_sessions.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        fieldnames = [
            "Session_ID",
            "Start_time",
            "User_intent",
            "Resolution_Status",
            "Final_Sentiment",
            "Handoff_to_agent",
            "total_messages",
            "conversation_duration",
            "Conversation_Feedback",
        ]

        # --- read existing rows, drop any old row for this session, and rewrite file ---
        existing_rows = []
        if os.path.exists(csv_path):
            with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("Session_ID") != session_id:
                        existing_rows.append(r)

        # add/replace current row for this session
        existing_rows.append(row)

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)

        # mark that export happened at least once
        return [
            SlotSet("final_sentiment", final_sentiment),
            SlotSet("export_done", True),
        ]
class ActionMarkFallback(Action):
    def name(self) -> Text:
        return "action_mark_fallback"

    def run(self, dispatcher, tracker, domain):
        current = tracker.get_slot("resolution_status")
        if not current or current == "unknown":
            return [SlotSet("resolution_status", "fallback")]
        return []
class ActionUpdateAddress(Action):
    def name(self) -> Text: 
        return "action_update_address"

    def run(self, dispatcher, tracker, domain):
        oid = tracker.get_slot("order_id")
        addr = tracker.get_slot("new_address")

        # Missing data → fallback + export
        if not oid or not addr:
            dispatcher.utter_message(text="I need both the order ID and the new address.")
            events: List[EventType] = [SlotSet("resolution_status", "fallback")]
            export_events = ActionExportConversationSummary().run(
                dispatcher, tracker, domain
            )
            events.extend(export_events)
            return events

        rec = ORDER_DB.get(oid)

        # Order not found → fallback + export
        if not rec:
            dispatcher.utter_message(text=f"Order {oid} not found.")
            events = clear_context_slots() + [SlotSet("resolution_status", "fallback")]
            export_events = ActionExportConversationSummary().run(
                dispatcher, tracker, domain
            )
            events.extend(export_events)
            return events

        # Shipped: cannot update → fallback + export
        if rec["status"].lower() == "shipped":
            dispatcher.utter_message(
                text=f"Order {oid} already shipped; I can’t change the address. Try carrier redirect."
            )
            events = clear_context_slots() + [SlotSet("resolution_status", "fallback")]
            export_events = ActionExportConversationSummary().run(
                dispatcher, tracker, domain
            )
            events.extend(export_events)
            return events

        # Successfully updated → resolved + export
        rec["address"] = addr
        dispatcher.utter_message(text=f"Address for {oid} updated to: {addr}.")
        events = clear_context_slots() + [SlotSet("resolution_status", "resolved")]
        export_events = ActionExportConversationSummary().run(
            dispatcher, tracker, domain
        )
        events.extend(export_events)
        return events
class ActionStoreFeedbackAndExport(Action):
    def name(self) -> Text:
        return "action_store_feedback_and_export"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[EventType]:

        feedback = tracker.get_slot("conversation_feedback")

        logger.info(
            "[feedback] session_id=%s feedback_slot=%r intent=%s",
            tracker.sender_id,
            feedback,
            tracker.latest_message.get("intent", {}).get("name"),
        )

        if feedback and feedback != "not_provided":
            dispatcher.utter_message(text="Thanks for your feedback! ")
        else:
            dispatcher.utter_message(text="Thanks for chatting with us! ")

        # IMPORTANT: this will now overwrite any previous row for this session
        export_events = ActionExportConversationSummary().run(
            dispatcher, tracker, domain
        )

        return export_events
######################DASHBOARD
class ActionPushStateToDashboard(Action):
    def name(self) -> Text:
        return "action_push_state_to_dashboard"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[Dict[Text, Any]]:

        sender_id = tracker.sender_id

        last_user_msg = next(
            (e.get("text") for e in reversed(tracker.events) if e.get("event") == "user"),
            None,
        )
        last_bot_msg = next(
            (e.get("text") for e in reversed(tracker.events) if e.get("event") == "bot"),
            None,
        )

        # ---- Slots (exactly as in your domain.yml) ----
        criticality_level = (tracker.get_slot("criticality_level") or "unknown").lower()
        criticality_score = tracker.get_slot("criticality_score")

        escalation = tracker.get_slot("escalation") or "none"
        conversation_escalation = tracker.get_slot("conversation_escalation") or "none"
        handoff_requested = bool(tracker.get_slot("handoff_requested"))

        # Conversation-level risk
        conversation_criticality = tracker.get_slot("conversation_criticality") or "unknown"
        conversation_criticality_score = tracker.get_slot("conversation_criticality_score")

        # Emotion/sentiment
        top_emotion = tracker.get_slot("top_emotion") or tracker.get_slot("overall_emotion") or "n/a"
        sentiment = tracker.get_slot("sentiment") or "n/a"

        # ---- Decide dashboard status color (green / black / red) ----
        if handoff_requested or escalation == "escalate_now" or conversation_escalation == "escalate_now":
            status = "red"
        elif criticality_level == "low":
            status = "green"
        else:
            status = "black"

        payload = {
            "session_id": sender_id,

            # For your list view (simple)
            "status": status,
            "criticality": criticality_level,
            "emotion": top_emotion,

            # Extra metadata (useful when agent clicks a session)
            "sentiment": sentiment,
            "criticality_score": criticality_score,
            "escalation": escalation,
            "conversation_criticality": conversation_criticality,
            "conversation_criticality_score": conversation_criticality_score,
            "conversation_escalation": conversation_escalation,
            "handoff_requested": handoff_requested,

            "last_user_message": last_user_msg,
            "last_bot_message": last_bot_msg,
            "updated_at": datetime.utcnow().isoformat() + "Z",

            # ── Phase 2: manager-level ABSA insights ──────────────────────────
            "absa_insights": _build_absa_insights(_absa_cache.get(sender_id, [])),
        }

        try:
            requests.post(
                f"{DASHBOARD_BACKEND_URL}/api/update_session",
                json=payload,
                timeout=1,
            )
        except Exception as e:
            print(f"[Dashboard] Failed to update session: {e}")

        return []