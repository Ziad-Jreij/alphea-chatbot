from __future__ import annotations
from typing import Any, Dict, List, Optional, Text
from rasa.shared.nlu.training_data.message import Message
from rasa.engine.graph import GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.constants import TEXT
from transformers import pipeline
import math
import re
import functools
import logging
from threading import Lock
import torch
import yaml
from pathlib import Path

def load_keywords():
    file = Path("../data/criticality_keywords.yml")
    if file.exists():
        data = yaml.safe_load(file.read_text())
        return data.get("high", []), data.get("medium", [])
    return [], []
# Shared model defaults (aligned with your config.yml)
_CRIT_MODEL_NAME = "facebook/bart-large-mnli"
_CRIT_LABELS = ["low urgency", "medium urgency", "high urgency"]
_CRIT_HYPOTHESIS = "This support request is {}."

_crit_lock = Lock()
_crit_pipe = None

# Deterministic mapping to escalation – same idea as in the component
ESCALATION_MAP = {"low": "none", "medium": "potential", "high": "escalate_now"}

# Emotion -> prior (log-odds) weights toward urgency (same structure you had)
EMOTION_WEIGHTS = {
    "anger":      {"high": +1.0, "medium": +0.5, "low": -0.3},
    "fear":       {"high": +1.2, "medium": +0.6, "low": -0.4},
    "disgust":    {"high": +0.6, "medium": +0.4, "low": -0.2},
    "sadness":    {"high": +0.4, "medium": +0.4, "low": -0.2},
    "neutral":    {"high": +0.0, "medium": +0.0, "low": +0.0},
    "joy":        {"high": -0.3, "medium": -0.2, "low": +0.2},
    "surprise":   {"high": +0.2, "medium": +0.1, "low": 0.0},
    "confusion":  {"high": +0.3, "medium": +0.2, "low": 0.0},
}

def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))

def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def ensure_crit_pipe(model_name: str = _CRIT_MODEL_NAME):
    """Lazy-init zero-shot pipeline. Can be used by NLU component and actions."""
    global _crit_pipe
    if _crit_pipe is None:
        with _crit_lock:
            if _crit_pipe is None:
                device = 0 if torch.cuda.is_available() else -1
                _crit_pipe = pipeline(
                    "zero-shot-classification",
                    model=model_name,
                    device=device,
                )
    return _crit_pipe       
def infer_criticality_for_text(
    text: str,
    emotion: Optional[str],
    *,
    model_name: str = _CRIT_MODEL_NAME,
    hypothesis_template: str = _CRIT_HYPOTHESIS,
    labels: Optional[List[str]] = None,
    boost_from_emotion: bool = True,
    heuristics_enabled: bool = True,
    high_keywords: Optional[List[str]] = None,
    medium_keywords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if labels is None:
        labels = _CRIT_LABELS

    if high_keywords is None:
        high_keywords = []
    if medium_keywords is None:
        medium_keywords = []

    pipe = ensure_crit_pipe(model_name)
    text = (text or "").strip()

    # 1) zero-shot
    zs = pipe(
        text,
        candidate_labels=labels,
        hypothesis_template=hypothesis_template,
        multi_label=False,
    )

    mapping = {
        "low urgency": "low",
        "medium urgency": "medium",
        "high urgency": "high",
    }
    raw = {mapping[l]: p for l, p in zip(zs["labels"], zs["scores"])}

    # 2) emotion prior
    fused = raw
    if boost_from_emotion and emotion:
        e = emotion.lower()
        weights = EMOTION_WEIGHTS.get(e, {"high": 0.0, "medium": 0.0, "low": 0.0})
        logits = {
            "low": _logit(raw["low"]) + weights["low"],
            "medium": _logit(raw["medium"]) + weights["medium"],
            "high": _logit(raw["high"]) + weights["high"],
        }
        probs = {k: _sigmoid(v) for k, v in logits.items()}
        total = sum(probs.values())
        fused = {k: v / total for k, v in probs.items()}

    # 3) keyword overrides (if enabled)
    if heuristics_enabled:
        text_l = text.lower()
        for k in high_keywords:
            if re.search(k, text_l, re.I):
                fused = {"low": 0.01, "medium": 0.09, "high": 0.90}
                break
        else:
            for k in medium_keywords:
                if re.search(k, text_l, re.I):
                    fused = {"low": 0.15, "medium": 0.70, "high": 0.15}
                    break

    # 4) pick argmax
    level = max(fused, key=fused.get)
    score = float(fused[level])
    escalation = ESCALATION_MAP[level]

    return {
        "criticality_level": level,
        "criticality_score": score,
        "escalation": escalation,
        "distribution": fused,
    }

@DefaultV1Recipe.register(
    component_types=[GraphComponent], is_trainable=False
)
class CriticalityZeroShotComponent(GraphComponent):
    def __init__(self, config: Dict[Text, Any]) -> None:
        self.model_name = config.get("model_name", "facebook/bart-large-mnli")
        self.hypothesis_template = config.get("hypothesis_template", "This request is {label}.")
        self.labels = config.get("labels", ["low urgency","medium urgency","high urgency"])
        self.boost_from_emotion = config.get("boost_from_emotion", True)
        self.heuristics_enabled = config.get("heuristics_enabled", True)
        file_high, file_medium = load_keywords()
        self.high_keywords = [re.compile(k, re.I) for k in file_high]
        self.medium_keywords = [re.compile(k, re.I) for k in file_medium]
        self.gpu = config.get("gpu", False)
        self.cache_enabled = config.get("cache", True)
        self._cache: Dict[str, Dict[str, Any]] = {}

        device = 0 if self.gpu else -1
        self.zs = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=device
        )

        self.emotion_weights = {
            "anger":      {"high": +1.0, "medium": +0.5, "low": -0.3},
            "fear":       {"high": +1.2, "medium": +0.6, "low": -0.4},
            "disgust":    {"high": +0.6, "medium": +0.4, "low": -0.2},
            "sadness":    {"high": +0.4, "medium": +0.4, "low": -0.2},
            "neutral":    {"high": +0.0, "medium": +0.0, "low": +0.0},
            "joy":        {"high": -0.3, "medium": -0.2, "low": +0.2},
            "surprise":   {"high": +0.2, "medium": +0.1, "low": 0.0},
            "confusion":  {"high": +0.3, "medium": +0.2, "low": 0.0},
        }

        # Deterministic mapping to escalation
        self.escalation_map = {"low": "none", "medium": "potential", "high": "escalate_now"}

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {}

    def _key(self, text: str, emotion: Optional[str]) -> str:
        return f"{emotion}|{text}" if emotion else text

    def _logit(self, p: float) -> float:
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))

    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def _apply_emotion_prior(self, scores: Dict[str, float], emotion: Optional[str]) -> Dict[str, float]:
        if not self.boost_from_emotion or not emotion:
            return scores
        e = emotion.lower()
        weights = self.emotion_weights.get(e, {"high": 0.0, "medium": 0.0, "low": 0.0})
        # Convert to logits, add priors, then back to probs, renormalize
        logits = {
            "low": self._logit(scores["low"]) + weights["low"],
            "medium": self._logit(scores["medium"]) + weights["medium"],
            "high": self._logit(scores["high"]) + weights["high"],
        }
        probs = {k: self._sigmoid(v) for k, v in logits.items()}
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}

    def _apply_keyword_overrides(self, text: str, scores: Dict[str, float]) -> Dict[str, float]:
        if not self.heuristics_enabled:
            return scores
        # Hard-raise if critical keywords are present
        for pat in self.high_keywords:
            if pat.search(text):
                return {"low": 0.01, "medium": 0.09, "high": 0.90}
        for pat in self.medium_keywords:
            if pat.search(text):
                # Nudge to medium if it's not already
                return {"low": 0.15, "medium": 0.70, "high": 0.15}
        return scores

    def process(self, messages: List[Message]) -> List[Message]:
        for m in messages:
            text = m.get(TEXT) or ""
            emotion = m.get("top_emotion")

            cache_key = self._key(text, emotion)
            if self.cache_enabled and cache_key in self._cache:
                result = self._cache[cache_key]
            else:
                result = infer_criticality_for_text(
                    text,
                    emotion,
                    model_name=self.model_name,
                    hypothesis_template=self.hypothesis_template,
                    labels=self.labels,
                    boost_from_emotion=self.boost_from_emotion,
                    heuristics_enabled=self.heuristics_enabled,
                    high_keywords=[p.pattern for p in self.high_keywords],
                    medium_keywords=[p.pattern for p in self.medium_keywords],
                )
                if self.cache_enabled:
                    self._cache[cache_key] = result

            level = result["criticality_level"]
            score = float(result["criticality_score"])
            escalation = result["escalation"]
            fused = result["distribution"]

            m.set("criticality_level", level)
            m.set("criticality_score", score)
            m.set("escalation", escalation)
            m.set("criticality_distribution", fused)

            entities = m.get("entities", [])
            entities.extend([
                {"entity": "criticality_level", "value": level},
                {"entity": "criticality_score", "value": float(score)},
                {"entity": "escalation", "value": escalation},
            ])
            m.set("entities", entities, add_to_output=True)

            logging.getLogger(__name__).info(
                "[criticality] text=%r emo=%s dist=%s -> level=%s score=%.2f escalation=%s",
                text, (emotion or "none"),
                {k: f"{v:.2f}" for k, v in fused.items()},
                level, score, escalation,
            )
        return messages


    def train(self, training_data: Any) -> None:
        # no training required
        return

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
    ) -> "CriticalityZeroShotComponent":
        return cls(config)
