from __future__ import annotations
from typing import Any, Dict, List, Optional, Text, Tuple

from pathlib import Path
import logging
import math
import re
from threading import Lock

import torch
import yaml
from transformers import pipeline
from rasa.engine.graph import GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword loading & compiled patterns (shared for NLU + actions)
# ---------------------------------------------------------------------------

_KEYWORD_LOCK = Lock()
_HIGH_KEYWORD_PATTERNS: List[Tuple[str, re.Pattern]] = []
_MEDIUM_KEYWORD_PATTERNS: List[Tuple[str, re.Pattern]] = []


def _load_keywords_from_file() -> Tuple[List[str], List[str]]:
    """Load raw keyword strings from YAML."""
    base_dir = Path(__file__).resolve().parent.parent  # project root
    file = base_dir / "data" / "criticality_keywords.yml"

    if not file.exists():
        logger.warning(
            "Criticality keywords file not found at %s; using empty keyword lists.",
            file,
        )
        return [], []

    data = yaml.safe_load(file.read_text()) or {}
    high_keywords = data.get("high_keywords", []) or []
    medium_keywords = data.get("medium_keywords", []) or []
    logger.debug(
        "[criticality] Loaded %d high and %d medium keyword patterns from %s",
        len(high_keywords),
        len(medium_keywords),
        file,
    )
    # make sure they are strings
    return [str(x) for x in high_keywords], [str(x) for x in medium_keywords]


def _ensure_keyword_patterns() -> None:
    """Compile keyword patterns once and cache them globally."""
    global _HIGH_KEYWORD_PATTERNS, _MEDIUM_KEYWORD_PATTERNS

    if _HIGH_KEYWORD_PATTERNS or _MEDIUM_KEYWORD_PATTERNS:
        return

    with _KEYWORD_LOCK:
        if _HIGH_KEYWORD_PATTERNS or _MEDIUM_KEYWORD_PATTERNS:
            return

        high_strings, medium_strings = _load_keywords_from_file()
        _HIGH_KEYWORD_PATTERNS = [
            (s, re.compile(s, re.I)) for s in high_strings if s.strip()
        ]
        _MEDIUM_KEYWORD_PATTERNS = [
            (s, re.compile(s, re.I)) for s in medium_strings if s.strip()
        ]

        logger.debug(
            "[criticality] Compiled keyword patterns: high=%d, medium=%d",
            len(_HIGH_KEYWORD_PATTERNS),
            len(_MEDIUM_KEYWORD_PATTERNS),
        )


def _apply_keyword_overrides_text(text: str) -> Optional[Dict[str, Any]]:
    """Apply keyword overrides to text. Returns a full result dict or None.

    Priority: high keywords first, then medium.
    """
    text = (text or "").strip()
    if not text:
        return None

    _ensure_keyword_patterns()
    lower = text.lower()

    # High first (guardrail)
    for level, patterns in (
        ("high", _HIGH_KEYWORD_PATTERNS),
        ("medium", _MEDIUM_KEYWORD_PATTERNS),
    ):
        for original, pat in patterns:
            if pat.search(lower):
                if level == "high":
                    distribution = {"low": 0.01, "medium": 0.09, "high": 0.90}
                else:
                    distribution = {"low": 0.15, "medium": 0.70, "high": 0.15}

                score = float(distribution[level])
                escalation = ESCALATION_MAP[level]

                logger.debug(
                    "[criticality] keyword override: text=%r level=%s score=%.2f esc=%s kw=%r",
                    text,
                    level,
                    score,
                    escalation,
                    original,
                )

                return {
                    "criticality_level": level,
                    "criticality_score": score,
                    "escalation": escalation,
                    "distribution": distribution,
                    "source": "keyword",
                    "matched_keyword": original,
                }

    return None
# ---------------------------------------------------------------------------
# Extra heuristic rules for tricky messages (safety, finance, contrast, etc.)
# ---------------------------------------------------------------------------

CONTRAST_MARKERS = [
    " but ",
    " even though ",
    " although ",
    " though ",
    " however ",
]

SAFETY_PATTERNS = [
    r"caught fire",
    r"on fire",
    r"burn(ing)? smell",
    r"smok(e|ing)",
    r"explod(ed|e|ing)",
    r"electric shock",
    r"shocked me",
    r"short circuit",
    r"sparking",
]

MEDICAL_PATTERNS = [
    r"wrong label",
    r"wrong medicine",
    r"wrong (dose|dosage)",
    r"expired (medicine|pill|drug)?",
    r"allergic reaction",
    r"severe side effects?",
]

FINANCE_PATTERNS = [
    r"transactions?\s+i didn[’']t make",
    r"charges?\s+i don[’']t recognize",
    r"unauthorized (charge|payment|transaction)",
    r"account (was )?hacked",
    r"identity theft",
    r"fraud(ulent)? charge",
    r"scam",
]

CORE_FUNCTION_PATTERNS = [
    r"keeps? freezing",
    r"keeps? crashing",
    r"crash(es|ed)? often",
    r"screen (keeps )?freez(ing|es)",
    r"won.?t start",
    r"won.?t turn on",
    r"not working",
    r"stopped working",
    r"doesn.?t work properly",
    r"battery dies? after",  # more general
]

FRUSTRATION_PATTERNS = [
    r"reported this (three|3|four|4|several|many) times",
    r"no one is helping me",
    r"no one helps me",
    r"no one is responding",
    r"no one ever replies",
    r"i feel ignored",
]

ALL_RULE_PATTERNS = (
    SAFETY_PATTERNS
    + MEDICAL_PATTERNS
    + FINANCE_PATTERNS
    + CORE_FUNCTION_PATTERNS
    + FRUSTRATION_PATTERNS
)

def _matches_any(text: str, patterns: List[str]) -> Optional[str]:
    """Return the first regex pattern that matches, or None."""
    for p in patterns:
        if re.search(p, text):
            return p
    return None


def _boost_level_prob(
    dist: Dict[str, float], level: str, target_prob: float
) -> Dict[str, float]:
    """Increase probability of `level` to at least `target_prob`, renormalizing others."""
    current = dist.get(level, 0.0)
    if current >= target_prob:
        return dist

    new_dist = dict(dist)
    new_dist[level] = target_prob

    # redistribute remaining mass across other levels proportionally
    other_keys = [k for k in dist.keys() if k != level]
    remaining_mass = 1.0 - target_prob
    current_other_sum = sum(dist[k] for k in other_keys) or 1e-9

    for k in other_keys:
        new_dist[k] = dist[k] * (remaining_mass / current_other_sum)

    return new_dist


def _apply_structural_rules(
    text: str, dist: Dict[str, float]
) -> Tuple[Dict[str, float], str, Optional[str]]:
    """Apply structural/contrast rules on top of model distribution.

    Returns: (new_distribution, source_suffix, matched_pattern)
    """
    lower = (text or "").lower()
    source_suffix = ""
    matched_pattern: Optional[str] = None

    # ---- Layer 0: hard safety / medical / finance overrides (force HIGH) ----
    for group, label in (
        (SAFETY_PATTERNS, "safety_rule"),
        (MEDICAL_PATTERNS, "medical_rule"),
        (FINANCE_PATTERNS, "finance_rule"),
    ):
        pat = _matches_any(lower, group)
        if pat:
            new_dist = _boost_level_prob(dist, "high", 0.90)
            return new_dist, f"+{label}", pat

    # ---- Layer 1: core function broken (bump to high if only medium) ----
    pat = _matches_any(lower, CORE_FUNCTION_PATTERNS)
    if pat:
        dist = _boost_level_prob(dist, "high", 0.78)
        source_suffix += "+core_rule"
        matched_pattern = pat

    # ---- Layer 2: contrast structures ("but", "even though", etc.) ----
    for marker in CONTRAST_MARKERS:
        if marker in lower:
            _, after = lower.split(marker, 1)
            pat2 = _matches_any(after, ALL_RULE_PATTERNS)
            if pat2:
                dist = _boost_level_prob(dist, "high", 0.80)
                source_suffix += "+contrast_rule"
                matched_pattern = matched_pattern or pat2
            break
    # ---- Layer 3: repeated complaints / feeling ignored (bump up) ----
    pat_frustration = _matches_any(lower, FRUSTRATION_PATTERNS)
    if pat_frustration:
        dist = _boost_level_prob(dist, "high", 0.78)
        source_suffix += "+frustration_rule"
        matched_pattern = matched_pattern or pat_frustration

    return dist, source_suffix, matched_pattern

# ---------------------------------------------------------------------------
# Shared model defaults (aligned with your config.yml)
# ---------------------------------------------------------------------------

_CRIT_MODEL_NAME = "facebook/bart-large-mnli"
_CRIT_LABELS = ["low urgency", "medium urgency", "high urgency"]
_CRIT_HYPOTHESIS = "This support request is {}."

_crit_lock = Lock()
_crit_pipe = None

# Deterministic mapping to escalation
ESCALATION_MAP = {"low":"none","medium":"potential","high":"escalate_now"}

# Emotion -> prior (log-odds) weights toward urgency
EMOTION_WEIGHTS: Dict[str, Dict[str, float]] = {
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
                logger.debug(
                    "Initializing zero-shot criticality pipeline (%s) on device %s",
                    model_name,
                    device,
                )
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
    # these two are now ignored; kept only for backward compatibility
    high_keywords: Optional[List[str]] = None,
    medium_keywords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Core inference logic, shared between the Rasa component and custom actions.

    Order:
      1) Keyword overrides (YAML) if `heuristics_enabled=True`.
      2) Zero-shot model.
      3) Optional emotion-based adjustment.
    """
    if labels is None:
        labels = _CRIT_LABELS

    text = (text or "").strip()

    # 0) Keyword overrides (guardrail, shared for NLU + actions)
    if heuristics_enabled:
        kw_res = _apply_keyword_overrides_text(text)
        if kw_res is not None:
            return kw_res

    pipe = ensure_crit_pipe(model_name)

    # 1) Zero-shot classification
    zs = pipe(
        text,
        candidate_labels=labels,
        hypothesis_template=hypothesis_template,
        multi_label=False,
    )

    # Map model labels to canonical "low/medium/high"
    mapping = {
        "low urgency": "low",
        "medium urgency": "medium",
        "high urgency": "high",
    }
    raw: Dict[str, float] = {}
    for label, score in zip(zs["labels"], zs["scores"]):
        key = mapping.get(label)
        if key is not None:
            raw[key] = float(score)

    # Ensure all keys exist; if not, fall back to a safe distribution
    if not all(k in raw for k in ("low", "medium", "high")):
        logger.warning(
            "Zero-shot labels %s did not map cleanly to {low, medium, high}. "
            "Falling back to neutral distribution.",
            zs["labels"],
        )
        raw = {"low": 0.8, "medium": 0.15, "high": 0.05}

    # 2) Emotion prior
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
        total = sum(probs.values()) or 1.0
        fused = {k: v / total for k, v in probs.items()}

    rule_suffix = ""
    rule_match: Optional[str] = None

    # 3) Structural rules on top of model+emotion (if enabled and no keyword override)
    if heuristics_enabled:
        fused, rule_suffix, rule_match = _apply_structural_rules(text, fused)

    # 4) Pick argmax
    level = max(fused, key=fused.get)
    score = float(fused[level])
    escalation = ESCALATION_MAP[level]

    source = "model" + rule_suffix

    # logger.info(
    #     "[criticality] text=%r -> level=%s score=%.2f esc=%s source=%s kw=%s",
    #     text,
    #     level,
    #     score,
    #     escalation,
    #     source,
    #     rule_match,
    # )

    return {
        "criticality_level": level,
        "criticality_score": score,
        "escalation": escalation,
        "distribution": fused,
        "source": source,
        "matched_keyword": rule_match,
    }

# ---------------------------------------------------------------------------
# Rasa Graph Component
# ---------------------------------------------------------------------------

@DefaultV1Recipe.register(component_types=[GraphComponent], is_trainable=False)
class CriticalityZeroShotComponent(GraphComponent):
    """Rasa NLU component that infers criticality + escalation from text + emotion."""

    def __init__(self, config: Dict[Text, Any]) -> None:
        # Configurable options (can be set in config.yml)
        self.model_name = config.get("model_name", _CRIT_MODEL_NAME)
        self.hypothesis_template = config.get(
            "hypothesis_template",
            _CRIT_HYPOTHESIS,
        )
        self.labels = config.get("labels", _CRIT_LABELS)
        self.boost_from_emotion = config.get("boost_from_emotion", True)
        self.heuristics_enabled = config.get("heuristics_enabled", True)
        self.cache_enabled = config.get("cache", True)

        # Simple in-memory cache: (emotion|text) -> inference result
        self._cache: Dict[str, Dict[str, Any]] = {}

        logger.debug(
            "Initialized CriticalityZeroShotComponent with model=%s, "
            "boost_from_emotion=%s, heuristics_enabled=%s, cache=%s",
            self.model_name,
            self.boost_from_emotion,
            self.heuristics_enabled,
            self.cache_enabled,
        )

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Default config values (can be overridden in config.yml)."""
        return {
            "model_name": _CRIT_MODEL_NAME,
            "hypothesis_template": _CRIT_HYPOTHESIS,
            "labels": _CRIT_LABELS,
            "boost_from_emotion": True,
            "heuristics_enabled": True,
            "cache": True,
        }

    def _key(self, text: str, emotion: Optional[str]) -> str:
        """Generate a stable cache key for (text, emotion)."""
        text = text or ""
        return f"{emotion}|{text}" if emotion else text

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
                )
                if self.cache_enabled:
                    self._cache[cache_key] = result

            level = result["criticality_level"]
            score = float(result["criticality_score"])
            escalation = result["escalation"]
            fused = result["distribution"]

            # Store on the message
            m.set("criticality_level", level)
            m.set("criticality_score", score)
            m.set("escalation", escalation)
            m.set("criticality_distribution", fused)
            m.set("criticality_source", result.get("source"))
            m.set("criticality_keyword", result.get("matched_keyword"))

            # Also expose as entities for rules/forms if needed
            entities = m.get("entities", [])
            entities.extend(
                [
                    {"entity": "criticality_level", "value": level},
                    {"entity": "criticality_score", "value": score},
                    {"entity": "escalation", "value": escalation},
                ]
            )
            m.set("entities", entities, add_to_output=True)

            logger.debug(
                "[criticality] text=%r emo=%s dist=%s -> level=%s score=%.2f escalation=%s source=%s kw=%s",
                text,
                (emotion or "none"),
                {k: f"{v:.2f}" for k, v in fused.items()},
                level,
                score,
                escalation,
                result.get("source"),
                result.get("matched_keyword"),
            )

        return messages

    def train(self, training_data: Any) -> None:
        # No training required (pure inference component)
        return

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
    ) -> "CriticalityZeroShotComponent":
        return cls(config)
