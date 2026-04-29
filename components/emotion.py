# components/emotion.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Text
from rasa.engine.graph import GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import TEXT
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from threading import Lock
import torch

MODEL_NAME = "joeddav/distilbert-base-uncased-go-emotions-student"

# ---------------------------------------------------------------------------
# Week 1 Task 1: 6 business-friendly emotion labels
# ---------------------------------------------------------------------------
# Raw GoEmotions labels (28) → collapsed into 6 business-readable buckets.
# Every label the model can produce is explicitly mapped here.
# Display names are what agents/dashboards see — no ML jargon.
#
# BUCKET      | DISPLAY NAME  | WHY BUSINESS-RELEVANT
# frustrated  | Frustrated    | Most escalation predictor; drives churn
# anxious     | Anxious       | Signals urgency / fear of loss
# disappointed| Disappointed  | Post-resolution risk; silent churn
# satisfied   | Satisfied     | Positive resolution signal
# confused    | Confused       | Knowledge-base gap signal
# neutral     | Neutral        | Baseline / informational tone
# ---------------------------------------------------------------------------

BUSINESS_EMOTION_DISPLAY: Dict[str, str] = {
    "frustrated":   "Frustrated",
    "anxious":      "Anxious",
    "disappointed": "Disappointed",
    "satisfied":    "Satisfied",
    "confused":     "Confused",
    "neutral":      "Neutral",
}

# Maps every raw GoEmotions label → business bucket
_RAW_TO_BUSINESS: Dict[str, str] = {
    # Frustrated bucket
    "anger":          "frustrated",
    "annoyance":      "frustrated",
    "disapproval":    "frustrated",
    "disgust":        "frustrated",

    # Anxious bucket
    "fear":           "anxious",
    "nervousness":    "anxious",

    # Disappointed bucket
    "disappointment": "disappointed",
    "remorse":        "disappointed",
    "grief":          "disappointed",
    "sadness":        "disappointed",
    "embarrassment":  "disappointed",

    # Satisfied bucket
    "admiration":     "satisfied",
    "amusement":      "satisfied",
    "approval":       "satisfied",
    "caring":         "satisfied",
    "gratitude":      "satisfied",
    "joy":            "satisfied",
    "love":           "satisfied",
    "optimism":       "satisfied",
    "pride":          "satisfied",
    "relief":         "satisfied",
    "excitement":     "satisfied",

    # Confused bucket
    "confusion":      "confused",
    "curiosity":      "confused",
    "surprise":       "confused",  # surprise often signals confusion in support

    # Neutral bucket
    "neutral":        "neutral",
    "realization":    "neutral",
    "desire":         "neutral",
}

# Business polarity (used downstream for coarse_from_label)
_BUSINESS_POLARITY: Dict[str, str] = {
    "frustrated":   "negative",
    "anxious":      "negative",
    "disappointed": "negative",
    "satisfied":    "positive",
    "confused":     "ambiguous",
    "neutral":      "neutral",
}

# Smoothing window: how many turns to look back (Task 2)
_SMOOTHING_WINDOW = 4

_init_lock = Lock()
_pipe: TextClassificationPipeline | None = None


def ensure_emotion_pipe() -> TextClassificationPipeline:
    global _pipe
    if _pipe is None:
        with _init_lock:
            if _pipe is None:
                device = 0 if torch.cuda.is_available() else -1
                tok = AutoTokenizer.from_pretrained(MODEL_NAME)
                mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
                _pipe = TextClassificationPipeline(
                    model=mdl, tokenizer=tok, top_k=None,
                    function_to_apply="sigmoid", device=device,
                )
    return _pipe


def map_to_business_emotion(raw_label: str) -> str:
    """Collapse a raw GoEmotions label to one of the 6 business buckets."""
    return _RAW_TO_BUSINESS.get((raw_label or "").lower(), "neutral")


def coarse_from_label(emotion_label: str) -> str:
    """Return positive / negative / neutral / ambiguous polarity."""
    bucket = map_to_business_emotion(emotion_label)
    return _BUSINESS_POLARITY.get(bucket, "ambiguous")


def top_emotion(text: str) -> tuple[str, list[tuple[str, float]]]:
    """Run the model and return (business_bucket, top3_raw_predictions)."""
    pipe = ensure_emotion_pipe()
    preds = pipe((text or "").strip())[0]
    top = max(preds, key=lambda x: x["score"])
    top3 = sorted(
        ((p["label"], float(p["score"])) for p in preds),
        key=lambda x: x[1], reverse=True,
    )[:3]
    business_label = map_to_business_emotion(top["label"])
    return business_label, top3


# ---------------------------------------------------------------------------
# Week 1 Task 2: Rolling emotion smoothing per session
# ---------------------------------------------------------------------------
# Instead of returning the raw single-turn prediction, we apply a recency-
# weighted rolling window over the last _SMOOTHING_WINDOW turns.
# This prevents a single ambiguous message from flipping the emotion label.
#
# Algorithm:
#   weight[i] = 1 / (rank + 1)  (most recent turn = rank 0, highest weight)
#   smoothed = argmax of weighted vote across window
#
# Returns the smoothed label and the reason string for explainability.
# ---------------------------------------------------------------------------

def smooth_emotion(current_label: str, history: List[str]) -> tuple[str, str]:
    """
    Apply recency-weighted smoothing over `history` + current label.

    Args:
        current_label: business bucket for the current turn
        history: list of business buckets, most recent last, max _SMOOTHING_WINDOW

    Returns:
        (smoothed_label, reason_code)
        reason_code: 'stable'|'smoothed_from_<prev>'|'new_signal'
    """
    window = (list(history) + [current_label])[-_SMOOTHING_WINDOW:]

    # Weighted vote (most recent = highest weight)
    weights: Dict[str, float] = {}
    for rank, label in enumerate(reversed(window)):
        w = 1.0 / (rank + 1)
        weights[label] = weights.get(label, 0.0) + w

    smoothed = max(weights, key=weights.get)

    # Build reason code
    prev = history[-1] if history else None
    if smoothed == current_label:
        reason = "stable" if (prev == current_label) else "new_signal"
    else:
        reason = f"smoothed_from_{current_label}_to_{smoothed}"

    return smoothed, reason


@DefaultV1Recipe.register(component_types=[GraphComponent], is_trainable=False)
class EmotionClassifierComponent(GraphComponent):

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {"smoothing_window": _SMOOTHING_WINDOW}

    @classmethod
    def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource):
        return cls(config)

    def __init__(self, config: Dict[Text, Any]) -> None:
        self.config = config

    def train(self, training_data: Any) -> None:
        return

    def process(self, messages: List[Message]) -> List[Message]:
        for m in messages:
            text = m.get(TEXT) or ""
            # Raw → business bucket
            business_label, top3 = top_emotion(text)
            m.set("top_emotion", business_label, add_to_output=True)
            m.set("top_emotion_display", BUSINESS_EMOTION_DISPLAY.get(business_label, business_label), add_to_output=True)
            # Smoothing is applied at the action layer (where session history lives)
            # NLU component sets the single-turn value; smoothing happens in actions.py
        return messages