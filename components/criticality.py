from __future__ import annotations
from typing import Any, Dict, List, Optional, Text, Tuple

from pathlib import Path
import logging
import math
import re
from threading import Lock

import os

import torch
import yaml
from transformers import pipeline
try:
    from rasa.engine.graph import GraphComponent
    from rasa.engine.recipes.default_recipe import DefaultV1Recipe
    from rasa.engine.storage.resource import Resource
    from rasa.engine.storage.storage import ModelStorage
    from rasa.shared.nlu.constants import TEXT
    from rasa.shared.nlu.training_data.message import Message
except ImportError:
    class GraphComponent:  # type: ignore[no-redef]
        pass
    class DefaultV1Recipe:  # type: ignore[no-redef]
        @classmethod
        def register(cls, **kwargs):
            return lambda f: f
    Resource = None  # type: ignore[assignment]
    ModelStorage = None  # type: ignore[assignment]
    Message = None  # type: ignore[assignment]
    TEXT = "text"

logger = logging.getLogger(__name__)

# Torch defaults to one thread per core; on many-core CPUs that thrashes on the
# small batches used here (tail latency of seconds). 8 threads measured best.
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", min(8, torch.get_num_threads()))))

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



# ---------------------------------------------------------------------------
# Week 2 Task 1: Realistic keyword guardrail hardening
# ---------------------------------------------------------------------------
# Problems fixed vs Week 1:
#
#  A) NEGATION FALSE POSITIVES
#     "my card was NOT stolen" was matching "stolen card" → now suppressed.
#     Negation window: if any negation word appears within _NEGATION_WINDOW_TOKENS
#     tokens before the match span, the keyword is considered negated and skipped.
#
#  B) HEARSAY / HYPOTHETICAL FALSE POSITIVES
#     "I heard this is a scam" / "wondering if this is fraud" → negation tokens
#     like "heard", "wondering", "if", "whether" now cancel the match.
#
#  C) SOURCE TAG PRECISION
#     Source now distinguishes "keyword_high" vs "keyword_medium" (not just
#     "keyword") so the audit trail identifies which list fired.
#
#  D) MATCHED SPAN CAPTURED
#     The actual text span that triggered the keyword is returned as
#     "matched_span" for audit logging and dashboard display.
# ---------------------------------------------------------------------------

_NEGATION_PREFIXES = frozenset({
    "not", "no", "never", "wasn't", "didn't", "don't", "haven't",
    "isn't", "aren't", "doesn't", "can't", "couldn't", "wouldn't",
    "won't", "without", "deny", "denying", "false", "fake",
    "supposedly", "apparently", "heard", "wondering",
    "whether", "ask", "asking", "maybe", "perhaps", "possibly",
})

_NEGATION_WINDOW_TOKENS = 5


def _is_negated(text_lower: str, match_start: int) -> bool:
    """Return True if the match appears to be negated or hypothetical."""
    prefix = text_lower[:match_start]
    tokens = prefix.split()[-_NEGATION_WINDOW_TOKENS:]
    return bool(set(tokens) & _NEGATION_PREFIXES)


def _apply_keyword_overrides_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Apply keyword guardrails to text. Returns a full result dict or None.

    Week 2 improvements:
    - Negation / hearsay guard cancels false-positive matches
    - Source tag distinguishes keyword_high vs keyword_medium
    - matched_span records the actual triggering text for the audit trail
    """
    text = (text or "").strip()
    if not text:
        return None

    _ensure_keyword_patterns()
    lower = text.lower()

    for level, patterns, source_tag in (
        ("high",   _HIGH_KEYWORD_PATTERNS,   "keyword_high"),
        ("medium", _MEDIUM_KEYWORD_PATTERNS, "keyword_medium"),
    ):
        for original, pat in patterns:
            m = pat.search(lower)
            if not m:
                continue

            # Negation / hearsay guard
            if _is_negated(lower, m.start()):
                logger.debug(
                    "[criticality] keyword NEGATED: kw=%r span=%r text=%r",
                    original, m.group(), text[:100],
                )
                continue

            distribution = (
                {"low": 0.01, "medium": 0.09, "high": 0.90}
                if level == "high"
                else {"low": 0.15, "medium": 0.70, "high": 0.15}
            )
            score = float(distribution[level])
            escalation = ESCALATION_MAP[level]

            logger.debug(
                "[criticality] keyword override: kw=%r span=%r level=%s score=%.2f",
                original, m.group(), level, score,
            )

            reason_code = build_reason_code(source_tag, original, level, score)
            explanation = explain_escalation({
                "criticality_level": level,
                "criticality_score": score,
                "source": source_tag,
                "matched_keyword": original,
                "emotion_used": None,
                "reason_code": reason_code,
            })

            return {
                "criticality_level": level,
                "criticality_score": score,
                "escalation": escalation,
                "distribution": distribution,
                "source": source_tag,
                "matched_keyword": original,
                "matched_span": m.group(),
                "reason_code": reason_code,
                "explanation": explanation,
                "emotion_used": None,
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
    # Device smoking / smoke coming out — not any mention ("I don't smoke").
    r"\bsmok(ing|ed)\b",
    r"smoke (is )?(coming|came|pouring) (out|from)",
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
    # The group used to be optional, so "my coupon expired yesterday" was HIGH.
    r"expired (medicines?|medications?|pills?|drugs?)",
    r"(medicines?|medications?|pills?|drugs?) (is|are|was|were) expired",
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
    # Require repetition/persistence markers to avoid single-report false positives
    r"keeps? (freezing|crashing|restarting|disconnecting)",
    r"crash(es|ed)? (often|repeatedly|again|constantly|every time)",
    r"screen (keeps )?freez(ing|es)",
    r"won[’']?t (start|turn on|boot|open|load)",
    # "not working" only with subject anchor (device/app/phone/system)
    r"(device|phone|app|system|laptop|tablet|router|printer)\s+(is\s+)?(not working|stopped working|doesn[’']?t work)",
    # Battery drain with specificity
    r"battery (dies?|drains?) (after|in) \d",
    # Complete failure phrases
    r"completely (dead|broken|stopped|unusable)",
    r"(won[’']?t|doesn[’']?t) (do anything|respond|work at all)",
]

_REPEAT_COUNT = r"(three|four|five|six|seven|eight|nine|ten|several|many|multiple|countless|\d+)"

FRUSTRATION_PATTERNS = [
    r"reported this (three|3|four|4|several|many) times",
    # Repeated unresolved contact: "contacted you three times", "the fourth
    # time I'm writing", "six emails", "called five times this week".
    rf"(contacted|called|emailed|written|wrote|messaged|chatted|complained|asked)"
    rf"( to)?( you| support| your team)?( about \w+( \w+)?)? {_REPEAT_COUNT} times",
    r"\b(third|fourth|fifth|sixth|\d+(st|nd|rd|th)) time (i|i'm|i am|i've|you|that|this)\b",
    rf"\b{_REPEAT_COUNT} (emails|calls|messages|tickets|complaints)\b",
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

def _matches_any(text: str, patterns: List[str], skip_negated: bool = False) -> Optional[str]:
    """Return the first regex pattern that matches, or None.

    With skip_negated, matches preceded by a negation/hearsay word ("no
    unauthorized charge", "I heard this is a scam") are ignored.
    """
    for p in patterns:
        for m in re.finditer(p, text):
            if not (skip_negated and _is_negated(text, m.start())):
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
    # Only the finance rules honour negation/hearsay: the window is crude, and
    # for safety/medical a missed escalation ("I can't believe it caught
    # fire") is worse than a false alarm.
    for group, label, skip_negated in (
        (SAFETY_PATTERNS, "safety_rule", False),
        (MEDICAL_PATTERNS, "medical_rule", False),
        (FINANCE_PATTERNS, "finance_rule", True),
    ):
        pat = _matches_any(lower, group, skip_negated=skip_negated)
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
# Week 1 Task 3: Three criticality levels with EXPLICIT, human-readable rules
# ---------------------------------------------------------------------------
#
#  LEVEL   | ESCALATION       | TRIGGERS (at least ONE must be true)
# ---------+------------------+------------------------------------------------------
#  HIGH    | escalate_now     | • keyword match in high_keywords YAML
#          |                  | • safety / medical / finance hard-override rule fires
#          |                  | • zero-shot model confidence ≥ 0.65 for "high urgency"
#          |                  | • frustration or core-function rule fires AND emotion
#          |                  |   is frustrated/anxious (emotion confirms distress)
# ---------+------------------+------------------------------------------------------
#  MEDIUM  | potential        | • keyword match in medium_keywords YAML
#          |                  | • zero-shot model confidence ≥ 0.45 for "medium urgency"
#          |                  | • core-function rule fires but emotion is neutral/calm
#          |                  | • contrast rule fires (e.g. "works fine but the app crashes")
# ---------+------------------+------------------------------------------------------
#  LOW     | none             | • none of the above match
#          |                  | • model confidence < 0.45 for any urgency band
#          |                  | • message is clearly informational / inquiry tone
# ---------------------------------------------------------------------------
#
# Week 1 Task 4: Every decision produces a structured REASON CODE
# ---------------------------------------------------------------------------
# Format:  "<source>:<detail>"
#
# source values:
#   keyword_high    – matched a high-priority YAML keyword
#   keyword_medium  – matched a medium-priority YAML keyword
#   rule_safety     – safety regex fired  (e.g. "on fire", "electric shock")
#   rule_medical    – medical regex fired  (e.g. "wrong medicine", "allergic reaction")
#   rule_finance    – finance regex fired  (e.g. "unauthorized charge", "account hacked")
#   rule_core       – core-function regex fired  (e.g. "keeps crashing", "won't start")
#   rule_frustration– frustration regex fired  (e.g. "no one is helping me")
#   rule_contrast   – contrast structure detected  (e.g. "… but it crashed")
#   model+emotion   – zero-shot model result, adjusted by emotion prior
#   model           – zero-shot model result only (emotion not available / neutral)
#
# detail: keyword/pattern that triggered, or "conf=0.72" for model paths
#
# This means ANY escalation decision can be explained in < 30 seconds.
# ---------------------------------------------------------------------------

_CRIT_MODEL_NAME = os.environ.get("CRITICALITY_MODEL", "MoritzLaurer/deberta-v3-base-zeroshot-v2.0")
_CRIT_LABELS = ["low urgency", "medium urgency", "high urgency"]
_CRIT_HYPOTHESIS = "This support request is {}."

# "risk" (default) scores the categories below independently; "urgency" is the
# original single softmax over low/medium/high urgency (kept for comparison).
CRITICALITY_STRATEGY = os.environ.get("CRITICALITY_STRATEGY", "risk")

# Tuned on the dev split of data/eval/criticality_benchmark.yml (2026-10);
# see eval_criticality.py for the numbers. Edit together with the benchmark.
RISK_HYPOTHESIS = "This message reports {}."
RISK_CATEGORIES: Dict[str, List[str]] = {
    "high": [
        "a safety hazard such as fire, smoke, sparks, overheating or an electric shock",
        "a physical injury or danger to a person or a child",
        "fraud, an unauthorized payment or a stolen card",
        "a hacked, compromised or taken-over account",
        "a privacy breach or exposed personal data",
        "a medical or health risk",
        # Was "a legal threat, a chargeback or a formal complaint", which scored
        # 0.7-0.9 on plain return requests ("I want to return my order").
        "a threat to take legal action",
    ],
    "medium": [
        "a late, lost or missing delivery",
        "a wrong, damaged or defective item",
        "a billing, refund or payment problem",
        "poor customer service or no reply from support",
        "a website, app or account access problem",
    ],
    "low": [
        "a general question or information request",
        "a compliment or a thank you",
        "a routine request to change an order or account",
    ],
}
RISK_HIGH_THRESHOLD = 0.70   # strongest risk category (after emotion prior) ≥ this → high

_crit_lock = Lock()
_crit_pipe = None

# Deterministic mapping to escalation (explicit — no magic numbers)
ESCALATION_MAP = {"low": "none", "medium": "potential", "high": "escalate_now"}

# Confidence thresholds that define the level boundaries (Task 3 rule: explicit)
HIGH_CONF_THRESHOLD   = 0.65   # model output ≥ this → high
MEDIUM_CONF_THRESHOLD = 0.45   # model output ≥ this → medium; below → low

# Emotion -> prior (log-odds) weights toward urgency
# Updated to use the 6 business-friendly emotion buckets (Task 1)
EMOTION_WEIGHTS: Dict[str, Dict[str, float]] = {
    # business buckets
    "frustrated":   {"high": +1.0, "medium": +0.5, "low": -0.3},
    "anxious":      {"high": +1.2, "medium": +0.6, "low": -0.4},
    "disappointed": {"high": +0.4, "medium": +0.4, "low": -0.2},
    "confused":     {"high": +0.3, "medium": +0.2, "low": 0.0},
    "neutral":      {"high": +0.0, "medium": +0.0, "low": +0.0},
    "satisfied":    {"high": -0.3, "medium": -0.2, "low": +0.2},
    # legacy raw labels kept for backward compatibility during transition
    "anger":        {"high": +1.0, "medium": +0.5, "low": -0.3},
    "fear":         {"high": +1.2, "medium": +0.6, "low": -0.4},
    "disgust":      {"high": +0.6, "medium": +0.4, "low": -0.2},
    "sadness":      {"high": +0.4, "medium": +0.4, "low": -0.2},
    "joy":          {"high": -0.3, "medium": -0.2, "low": +0.2},
    "surprise":     {"high": +0.2, "medium": +0.1, "low": 0.0},
    "confusion":    {"high": +0.3, "medium": +0.2, "low": 0.0},
}

# ---------------------------------------------------------------------------
# Reason-code builder (Task 4)
# ---------------------------------------------------------------------------

def build_reason_code(source: str, detail: Optional[str], level: str, score: float) -> str:
    """
    Produce a human-readable reason code for any criticality decision.

    Examples:
      "keyword_high:stolen card"
      "rule_safety:caught fire"
      "model+emotion:conf=0.71,emotion=frustrated"
      "model:conf=0.52"
    """
    base = source.replace("+", "_and_")
    if detail:
        return f"{base}:{detail}"
    return f"{base}:conf={score:.2f}"


def explain_escalation(result: Dict[str, Any]) -> str:
    """
    Return a single English sentence that explains an escalation decision.
    Designed to let any team member understand a decision in < 30 seconds.

    Example outputs:
      "Escalated to HIGH because a high-priority keyword matched: 'unauthorized charge'."
      "Escalated to HIGH because a safety rule fired: pattern 'caught fire'."
      "Escalated to MEDIUM because the model rated urgency at 0.58 (threshold 0.45), boosted by Frustrated emotion."
      "No escalation (LOW): the model rated this LOW with confidence 0.78; no keywords or rules matched."
    """
    level     = (result.get("criticality_level") or "low").upper()
    source    = result.get("source") or "model"
    score     = float(result.get("criticality_score") or 0.0)
    keyword   = result.get("matched_keyword")
    emotion   = result.get("emotion_used")

    if "keyword_high" in source or (source == "keyword" and result.get("criticality_level") == "high"):
        return f"Escalated to {level} because a high-priority keyword matched: '{keyword}'."

    if "keyword_medium" in source or (source == "keyword" and result.get("criticality_level") == "medium"):
        return f"Raised to {level} because a medium-priority keyword matched: '{keyword}'."

    if "safety_rule" in source:
        return f"Escalated to {level} because a safety rule fired: pattern '{keyword}'."
    if "medical_rule" in source:
        return f"Escalated to {level} because a medical-risk rule fired: pattern '{keyword}'."
    if "finance_rule" in source:
        return f"Escalated to {level} because a financial-risk rule fired: pattern '{keyword}'."
    if "core_rule" in source:
        return f"Raised to {level} because a core-function failure rule fired: pattern '{keyword}'."
    if "frustration_rule" in source:
        return f"Escalated to {level} because a repeated-frustration rule fired: pattern '{keyword}'."
    if "contrast_rule" in source:
        return f"Raised to {level} because a contrast structure was detected (problem after 'but/however'): pattern '{keyword}'."

    # Model path
    emo_part = f", boosted by '{emotion}' emotion" if emotion and emotion not in ("neutral", "none") else ""
    risk_category = result.get("risk_category")
    if source.startswith("risk_model") and level == "HIGH" and risk_category:
        return f"Escalated to HIGH because the model detected {risk_category} (score {score:.2f}){emo_part}."
    if source.startswith("risk_model") and level == "MEDIUM":
        return f"Raised to MEDIUM: the model sees a customer problem without a safety, fraud or legal risk{emo_part}."
    thresh = HIGH_CONF_THRESHOLD if level == "HIGH" else MEDIUM_CONF_THRESHOLD
    if level in ("HIGH", "MEDIUM"):
        return (
            f"Escalated to {level} because the model rated urgency at {score:.2f} "
            f"(threshold {thresh:.2f}){emo_part}."
        )
    return (
        f"No escalation ({level}): the model rated this {level} with confidence {score:.2f}; "
        f"no keywords or rules matched."
    )


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _resolve_device() -> int:
    """Return the torch device index based on the GPU_ENABLED env var.

    GPU_ENABLED=auto  (default) — use GPU when CUDA is available, otherwise CPU.
    GPU_ENABLED=true            — require GPU; raise at startup if none found.
    GPU_ENABLED=false           — always use CPU regardless of hardware.
    """
    setting = os.environ.get("GPU_ENABLED", "auto").lower().strip()
    if setting == "false":
        return -1
    if setting == "true":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU_ENABLED=true but no CUDA GPU was detected. "
                "Either install the CUDA drivers or set GPU_ENABLED=auto."
            )
        return 0
    # "auto"
    return 0 if torch.cuda.is_available() else -1


def ensure_crit_pipe(model_name: str = _CRIT_MODEL_NAME):
    """Lazy-init zero-shot pipeline. Can be used by NLU component and actions."""
    global _crit_pipe
    if _crit_pipe is None:
        with _crit_lock:
            if _crit_pipe is None:
                device = _resolve_device()
                logger.debug(
                    "Initializing zero-shot criticality pipeline (%s) on device %s "
                    "(GPU_ENABLED=%s)",
                    model_name,
                    device,
                    os.environ.get("GPU_ENABLED", "auto"),
                )
                _crit_pipe = pipeline(
                    "zero-shot-classification",
                    model=model_name,
                    device=device,
                )
    return _crit_pipe


def _emotion_weights(emotion: Optional[str]) -> Dict[str, float]:
    """Log-odds prior for an emotion label (raw GoEmotions or business bucket).

    The action server passes raw labels ("annoyance", "nervousness"), most of
    which are not keys of EMOTION_WEIGHTS; fall back to their business bucket
    so the prior is not silently skipped.
    """
    zero = {"high": 0.0, "medium": 0.0, "low": 0.0}
    if not emotion:
        return zero
    e = emotion.lower()
    if e in EMOTION_WEIGHTS:
        return EMOTION_WEIGHTS[e]
    from components.emotion import map_to_business_emotion
    return EMOTION_WEIGHTS.get(map_to_business_emotion(e), zero)


def _shift(p: float, weight: float) -> float:
    return _sigmoid(_logit(p) + weight)


def _urgency_distribution(pipe, text, labels, hypothesis_template, weights) -> Dict[str, float]:
    """Legacy strategy: one softmax over "low/medium/high urgency"."""
    zs = pipe(text, candidate_labels=labels, hypothesis_template=hypothesis_template, multi_label=False)
    mapping = {"low urgency": "low", "medium urgency": "medium", "high urgency": "high"}
    raw = {mapping[l]: float(sc) for l, sc in zip(zs["labels"], zs["scores"]) if l in mapping}
    if not all(k in raw for k in ("low", "medium", "high")):
        logger.warning("Zero-shot labels %s did not map to low/medium/high; using a neutral distribution.",
                       zs["labels"])
        raw = {"low": 0.8, "medium": 0.15, "high": 0.05}
    probs = {k: _shift(v, weights[k]) for k, v in raw.items()}
    total = sum(probs.values()) or 1.0
    return {k: v / total for k, v in probs.items()}


def _risk_distribution(pipe, text, weights) -> Tuple[Dict[str, float], str]:
    """Risk strategy: every category is scored independently (multi-label).

    HIGH probability is the strongest risk category, so a safety or fraud
    report escalates on its own merit instead of having to out-score "a
    defective item" in a single ranking. The remaining mass is split between
    medium and low by the strongest problem vs routine category.
    """
    labels = [label for group in RISK_CATEGORIES.values() for label in group]
    # One batched forward pass for all categories: ~190 ms vs ~550+ ms on CPU.
    zs = pipe(text, candidate_labels=labels, hypothesis_template=RISK_HYPOTHESIS, multi_label=True,
              batch_size=len(labels))
    scores = dict(zip(zs["labels"], zs["scores"]))
    top_risk = max(RISK_CATEGORIES["high"], key=lambda label: scores[label])

    risk    = _shift(scores[top_risk], weights["high"])
    problem = _shift(max(scores[label] for label in RISK_CATEGORIES["medium"]), weights["medium"])
    routine = _shift(max(scores[label] for label in RISK_CATEGORIES["low"]), weights["low"])
    medium_share = problem / ((problem + routine) or 1.0)
    dist = {"high": risk, "medium": (1 - risk) * medium_share, "low": (1 - risk) * (1 - medium_share)}
    return dist, top_risk


def infer_criticality_for_text(
    text: str,
    emotion: Optional[str],
    *,
    model_name: str = _CRIT_MODEL_NAME,
    hypothesis_template: str = _CRIT_HYPOTHESIS,
    labels: Optional[List[str]] = None,
    boost_from_emotion: bool = True,
    heuristics_enabled: bool = True,
    strategy: Optional[str] = None,
    # these two are now ignored; kept only for backward compatibility
    high_keywords: Optional[List[str]] = None,
    medium_keywords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Core inference logic, shared between the Rasa component and custom actions.

    Order:
      1) High-priority keywords (YAML) decide immediately; a medium keyword
         sets a floor of MEDIUM but no longer blocks a HIGH from the model
         or rules ("charged twice and I need this fixed immediately!").
      2) Zero-shot model ("risk" categories by default, or legacy "urgency").
      3) Emotion prior (log-odds shift).
      4) Structural safety / medical / finance / frustration rules.
    """
    strategy = (strategy or CRITICALITY_STRATEGY).lower()
    text = (text or "").strip()

    medium_floor: Optional[Dict[str, Any]] = None
    if heuristics_enabled:
        kw_res = _apply_keyword_overrides_text(text)
        if kw_res is not None:
            if kw_res["criticality_level"] == "high":
                return kw_res
            medium_floor = kw_res

    pipe = ensure_crit_pipe(model_name)
    weights = _emotion_weights(emotion) if boost_from_emotion else _emotion_weights(None)

    risk_category: Optional[str] = None
    if strategy == "urgency":
        fused = _urgency_distribution(pipe, text, labels or _CRIT_LABELS, hypothesis_template, weights)
        source = "model"
    else:
        fused, risk_category = _risk_distribution(pipe, text, weights)
        source = "risk_model"

    rule_suffix = ""
    rule_match: Optional[str] = None
    if heuristics_enabled:
        fused, rule_suffix, rule_match = _apply_structural_rules(text, fused)
    source += rule_suffix

    # Explicit level thresholds (Task 3)
    if strategy == "urgency":
        raw_argmax = max(fused, key=fused.get)
        if fused["high"] >= HIGH_CONF_THRESHOLD:
            level = "high"
        elif fused["medium"] >= MEDIUM_CONF_THRESHOLD or raw_argmax == "medium":
            level = "medium"
        else:
            level = "low"
    else:
        if fused["high"] >= RISK_HIGH_THRESHOLD:
            level = "high"
        else:
            level = "medium" if fused["medium"] >= fused["low"] else "low"

    matched = rule_match
    if medium_floor is not None and level == "low":
        level = "medium"
        source = "keyword_medium+" + source
        matched = medium_floor["matched_keyword"]

    score      = float(fused[level])
    escalation = ESCALATION_MAP[level]

    # Reason code + English explanation (Task 4)
    if matched:
        reason_detail = matched
    elif level == "high" and risk_category and not rule_suffix:
        reason_detail = f"{risk_category},conf={score:.2f}"
    elif emotion:
        reason_detail = f"conf={score:.2f},emotion={emotion}"
    else:
        reason_detail = f"conf={score:.2f}"

    reason_code = build_reason_code(source, reason_detail, level, score)
    explanation = explain_escalation({
        "criticality_level": level,
        "criticality_score": score,
        "source": source,
        "matched_keyword": matched,
        "risk_category": risk_category,
        "emotion_used": emotion,
        "reason_code": reason_code,
    })

    logger.info("[criticality] %s | reason=%s | text=%r", explanation, reason_code, text[:120])

    return {
        "criticality_level": level,
        "criticality_score": score,
        "escalation": escalation,
        "distribution": fused,
        "source": source,
        "matched_keyword": matched,
        "risk_category": risk_category,
        "reason_code": reason_code,   # NEW – Task 4
        "explanation": explanation,   # NEW – Task 4
        "emotion_used": emotion,      # NEW – transparency
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
        self._cache_max_size = config.get("cache_max_size", 512)

        # LRU cache capped at cache_max_size entries (OrderedDict, oldest evicted first)
        from collections import OrderedDict
        self._cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

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
                self._cache.move_to_end(cache_key)
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
                    if len(self._cache) > self._cache_max_size:
                        self._cache.popitem(last=False)  # evict oldest

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