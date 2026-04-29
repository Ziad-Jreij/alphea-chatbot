# components/absa.py
# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 – ABSA v1: Lightweight, Explainable Aspect-Based Sentiment Analysis
# ─────────────────────────────────────────────────────────────────────────────
#
# Architecture
# ────────────
#  1. Exact term match  (ASPECT_SPECS term lists)         → confidence anchor
#  2. Embedding similarity  (all-MiniLM-L6-v2, CPU-only)  → catches paraphrases
#     Only fires when exact match produces zero aspects.
#  3. Opinion scoring
#     • Seed-phrase embedding match (replaces bloated POS/NEG/NEU lists)
#     • Negation window detection
#  4. Confidence formula
#     base_weight × sentence_distance_decay × emotion_alignment_bonus
#  5. Business aspect roll-up
#     Every fine-grained aspect is tagged to one of 7 canonical business aspects
#     that managers actually ask about.
#
# No ML fine-tuning.  The sentence-transformer model is used inference-only.
# Graceful degradation: if sentence-transformers is not installed the module
# falls back silently to exact-match-only mode.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import math
import re
from threading import Lock
from typing import Dict, List, Optional, Tuple

from components.emotion import coarse_from_label  # global sentiment fallback

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# 1.  BUSINESS ASPECT TAXONOMY  (7 canonical manager-level aspects)
# ═════════════════════════════════════════════════════════════════════════════
# These are what appear in the manager dashboard — clear, jargon-free labels.
# Fine-grained sub-aspects (ASPECT_SPECS) map to exactly ONE business aspect.

BUSINESS_ASPECTS: Dict[str, str] = {
    # internal_key          : display label
    "customer_support":     "Customer Support",
    "product_quality":      "Product Quality",
    "delivery_experience":  "Delivery Experience",
    "digital_experience":   "Digital Experience",
    "pricing_and_value":    "Pricing & Value",
    "security_and_trust":   "Security & Trust",
    "return_and_refund":    "Returns & Refunds",
    "overall_experience":   "Overall Experience",   # synthetic fallback only
}

# ═════════════════════════════════════════════════════════════════════════════
# 2.  ASPECT SPECS
#     Each entry now carries:
#       name        – canonical fine-grained aspect name (unchanged)
#       terms       – surface term list for exact matching
#       category    – technical grouping (unchanged, kept for CSV compat.)
#       business    – which of the 7 business aspects this rolls up to
#       anchors     – 2-4 representative phrases used for embedding similarity
# ═════════════════════════════════════════════════════════════════════════════

ASPECT_SPECS: List[Dict] = [
    # ── Security / fraud ────────────────────────────────────────────────────
    {
        "name": "account security",
        "terms": [
            "account security", "suspicious transactions", "unauthorized transactions",
            "unauthorized charges", "fraudulent transaction", "fraudulent charge",
            "transactions i didn't make", "transactions i did not make",
            "charges i don't recognize", "charges i do not recognize",
            "card was used without my permission",
        ],
        "category": "Security",
        "business": "security_and_trust",
        "anchors": [
            "unauthorized transaction on my account",
            "someone used my card without permission",
            "suspicious activity detected",
        ],
    },
    {
        "name": "login",
        "terms": [
            "login", "log in", "sign in", "sign-in", "log-in",
            "logged in", "can't log in", "login issues",
        ],
        "category": "Security",
        "business": "security_and_trust",
        "anchors": [
            "I cannot log into my account",
            "login is not working",
        ],
    },
    {
        "name": "two-factor authentication",
        "terms": [
            "two-factor authentication", "two factor authentication", "2fa",
            "two-step verification", "2-step verification",
            "verification code", "authenticator app",
        ],
        "category": "Security",
        "business": "security_and_trust",
        "anchors": [
            "two-factor authentication issue",
            "verification code not arriving",
        ],
    },
    {
        "name": "fraud alert system",
        "terms": [
            "fraud alert system", "fraud alerts", "fraud alert",
            "suspicious activity alerts", "fraud detection",
            "blocked my card for fraud",
        ],
        "category": "Security",
        "business": "security_and_trust",
        "anchors": [
            "fraud alert on my account",
            "card blocked for suspicious activity",
        ],
    },

    # ── Product quality & reliability ────────────────────────────────────────
    {
        "name": "safety & defects",
        "terms": [
            "safety", "safe to use", "unsafe", "dangerous", "fire hazard",
            "caught fire", "on fire", "burned", "smelled like smoke",
            "smell of smoke", "leaking water", "leaking everywhere",
            "leaks everywhere", "water everywhere", "leaking", "leak",
            "electric shock", "gave me a shock", "short circuit",
        ],
        "category": "Reliability",
        "business": "product_quality",
        "anchors": [
            "the product is dangerous and caught fire",
            "device is leaking and unsafe to use",
            "electric shock from the product",
        ],
    },
    {
        "name": "product quality",
        "terms": [
            "product quality", "quality", "overall quality", "build quality",
            "build", "durability", "durable", "sturdy", "well built",
            "poorly built", "cheap materials", "materials", "material quality",
            "feels cheap", "feels solid",
        ],
        "category": "Features",
        "business": "product_quality",
        "anchors": [
            "the build quality feels cheap",
            "this product is well made and durable",
            "the materials feel flimsy",
        ],
    },
    {
        "name": "reliability",
        "terms": [
            "reliability", "reliable", "unreliable", "stability",
            "stable", "unstable", "uptime",
        ],
        "category": "Reliability",
        "business": "product_quality",
        "anchors": [
            "the service keeps going down",
            "the product is unreliable",
        ],
    },

    # ── Performance & features ───────────────────────────────────────────────
    {
        "name": "battery life",
        "terms": [
            "battery life", "battery", "battery duration", "battery performance",
            "battery power", "battery drains", "battery drain", "battery dies fast",
            "battery keeps dying", "charge lasts", "one charge", "holds a charge",
        ],
        "category": "Performance",
        "business": "product_quality",
        "anchors": [
            "the battery drains really fast",
            "battery life is terrible",
            "charge lasts all day",
        ],
    },
    {
        "name": "camera",
        "terms": [
            "camera", "photo quality", "photos", "pictures", "pics",
            "image quality", "takes great photos", "takes bad photos",
            "selfie camera", "front camera", "rear camera",
        ],
        "category": "Features",
        "business": "product_quality",
        "anchors": [
            "the camera takes blurry photos",
            "photo quality is excellent",
        ],
    },
    {
        "name": "performance",
        "terms": [
            "performance", "speed", "fast", "snappy", "smooth", "slow",
            "lag", "laggy", "lagging", "freezes", "keeps freezing", "stutters",
            "loading time", "load time", "runs well", "runs poorly",
            "responsive", "responsiveness",
        ],
        "category": "Performance",
        "business": "product_quality",
        "anchors": [
            "the app is very slow and laggy",
            "it freezes constantly",
            "runs smoothly and responsively",
        ],
    },
    {
        "name": "software update",
        "terms": [
            "software update", "app update", "system update",
            "latest version", "new version", "update broke", "after the update",
        ],
        "category": "Features",
        "business": "product_quality",
        "anchors": [
            "the latest update broke everything",
            "system update caused problems",
        ],
    },
    {
        "name": "internet speed",
        "terms": [
            "internet speed", "speed of my internet", "slow internet",
            "connection speed", "bandwidth", "slow connection", "fast connection",
            "download speed", "upload speed",
        ],
        "category": "Performance",
        "business": "product_quality",
        "anchors": [
            "my internet connection is very slow",
            "download speed is terrible",
        ],
    },
    {
        "name": "main feature",
        "terms": [
            "main feature", "key feature", "core feature", "primary feature",
            "the feature i needed", "feature i wanted",
        ],
        "category": "Features",
        "business": "product_quality",
        "anchors": [
            "the main feature is missing",
            "the key feature does not work",
        ],
    },

    # ── Usability / UX ──────────────────────────────────────────────────────
    {
        "name": "usability",
        "terms": [
            "usability", "ease of use", "easy to use", "simple to use",
            "user friendly", "user-friendly", "intuitive", "not intuitive",
            "hard to use", "difficult to use", "complicated to use",
            "confusing to use", "navigation", "hard to navigate",
            "easy to navigate", "interface is confusing", "UI is confusing",
            "user interface", "interface", "ui", "ui design",
        ],
        "category": "Usability",
        "business": "digital_experience",
        "anchors": [
            "the app is hard to navigate",
            "the interface is confusing and not intuitive",
            "very easy to use and straightforward",
        ],
    },
    {
        "name": "size guide",
        "terms": [
            "size guide", "sizing guide", "size chart", "sizing chart",
            "fit guide", "didn't match the size chart", "sizes are off",
        ],
        "category": "Usability",
        "business": "product_quality",
        "anchors": [
            "the size guide is inaccurate",
            "sizing chart does not match the actual product",
        ],
    },
    {
        "name": "design",
        "terms": [
            "design", "look", "looks", "appearance", "style", "aesthetic",
            "aesthetics", "visual design", "color", "colour", "shape",
            "sleek", "modern look", "ugly",
        ],
        "category": "Aesthetics",
        "business": "product_quality",
        "anchors": [
            "the design looks outdated and ugly",
            "the product has a beautiful modern design",
        ],
    },

    # ── Pricing & value ─────────────────────────────────────────────────────
    {
        "name": "price",
        "terms": [
            "price", "prices", "cost", "expensive", "too expensive",
            "overpriced", "cheap", "affordable", "pricing", "price point",
            "price tag",
        ],
        "category": "Cost",
        "business": "pricing_and_value",
        "anchors": [
            "the price is too expensive",
            "very affordable for the quality",
            "overpriced compared to competitors",
        ],
    },
    {
        "name": "value for money",
        "terms": [
            "value for money", "good value", "great value", "poor value",
            "not worth it", "worth it", "worth the money", "worth the price",
            "worth every penny", "bang for the buck",
        ],
        "category": "Cost",
        "business": "pricing_and_value",
        "anchors": [
            "not worth the money",
            "excellent value for the price",
        ],
    },
    {
        "name": "billing",
        "terms": [
            "billing", "bill", "bills", "charge", "charges", "extra charge",
            "charged twice", "wrong charge", "charge on my statement",
            "statement error", "billing issue", "billing problems",
        ],
        "category": "Cost",
        "business": "pricing_and_value",
        "anchors": [
            "I was charged twice on my bill",
            "incorrect charge on my statement",
            "billing error on my account",
        ],
    },
    {
        "name": "fees",
        "terms": [
            "fees", "fee", "atm fees", "hidden fees", "extra fees",
            "service fee", "maintenance fee", "late fee",
            "return shipping fee", "shipping fee", "return fee",
        ],
        "category": "Cost",
        "business": "pricing_and_value",
        "anchors": [
            "there are hidden fees I was not told about",
            "the service fee is too high",
        ],
    },
    {
        "name": "loan terms",
        "terms": [
            "loan terms", "loan conditions", "interest rate", "interest rates",
            "loan agreement", "repayment terms", "installments",
        ],
        "category": "Cost",
        "business": "pricing_and_value",
        "anchors": [
            "the interest rate on the loan is very high",
            "repayment terms are unfair",
        ],
    },

    # ── Delivery & logistics ─────────────────────────────────────────────────
    {
        "name": "delivery & shipping",
        "terms": [
            "delivery", "shipping", "shipped", "delivered",
            "shipping process", "delivery service", "courier",
            "package delivery", "shipping company",
        ],
        "category": "Logistics",
        "business": "delivery_experience",
        "anchors": [
            "my package was not delivered",
            "shipping took a very long time",
        ],
    },
    {
        "name": "delivery time",
        "terms": [
            "delivery time", "shipping time", "time to deliver",
            "arrived on time", "arrived late", "came late",
            "took forever to arrive", "shipping took forever",
            "shipping took long", "delayed delivery", "delivery delay",
            "fast delivery", "quick delivery",
        ],
        "category": "Logistics",
        "business": "delivery_experience",
        "anchors": [
            "the delivery was very late",
            "arrived much faster than expected",
            "took forever to arrive",
        ],
    },
    {
        "name": "packaging",
        "terms": [
            "packaging", "package", "the box", "box was damaged",
            "well packaged", "poorly packaged", "damaged packaging", "unboxing",
        ],
        "category": "Logistics",
        "business": "delivery_experience",
        "anchors": [
            "the packaging was damaged when it arrived",
            "product was well packaged and protected",
        ],
    },
    {
        "name": "delivery tracking",
        "terms": [
            "delivery tracker", "delivery tracking", "tracking",
            "track my delivery", "tracking page", "tracking link",
        ],
        "category": "Digital_Experience",
        "business": "delivery_experience",
        "anchors": [
            "the tracking link is not working",
            "I cannot track where my package is",
        ],
    },

    # ── Digital experience ───────────────────────────────────────────────────
    {
        "name": "app experience",
        "terms": [
            "app", "the app", "application", "mobile app", "phone app",
            "the app experience", "in the app", "app interface",
            "app design", "app layout",
        ],
        "category": "Digital_Experience",
        "business": "digital_experience",
        "anchors": [
            "the mobile app keeps crashing",
            "the app interface is confusing",
            "the app works perfectly",
        ],
    },
    {
        "name": "website experience",
        "terms": [
            "website", "site", "web site", "webpage", "web page",
            "on your site", "on the website", "online portal",
        ],
        "category": "Digital_Experience",
        "business": "digital_experience",
        "anchors": [
            "the website is down",
            "your online portal is very slow",
        ],
    },
    {
        "name": "order page",
        "terms": [
            "order page", "checkout page", "checkout", "payment page",
            "cart page", "shopping cart", "couldn't check out",
        ],
        "category": "Digital_Experience",
        "business": "digital_experience",
        "anchors": [
            "I cannot complete checkout on your site",
            "the payment page keeps crashing",
        ],
    },
    {
        "name": "online application process",
        "terms": [
            "online application process", "online application",
            "applied online", "application form online", "online form",
        ],
        "category": "Digital_Experience",
        "business": "digital_experience",
        "anchors": [
            "the online application form is broken",
            "I could not complete the application online",
        ],
    },

    # ── Customer support & service ───────────────────────────────────────────
    {
        "name": "customer support",
        "terms": [
            "customer support", "support team", "support", "customer service",
            "service agent", "agent on chat", "helpdesk", "help desk",
            "call center", "call centre", "hotline",
        ],
        "category": "Service",
        "business": "customer_support",
        "anchors": [
            "customer support was rude and unhelpful",
            "the support team resolved my issue quickly",
            "I could not reach anyone at customer service",
        ],
    },
    {
        "name": "branch staff",
        "terms": [
            "branch staff", "staff at the branch", "staff in the branch",
            "people at the branch", "bank staff", "teller", "tellers",
        ],
        "category": "Service",
        "business": "customer_support",
        "anchors": [
            "the staff at the branch were very rude",
            "the bank teller was helpful and friendly",
        ],
    },
    {
        "name": "response time",
        "terms": [
            "response time", "reply time", "responded quickly",
            "responded fast", "slow to respond", "took too long to respond",
            "quick response", "fast response", "no one replied",
        ],
        "category": "Service",
        "business": "customer_support",
        "anchors": [
            "nobody replied to my message for days",
            "they responded very quickly",
        ],
    },
    {
        "name": "card replacement",
        "terms": [
            "card replacement", "card was replaced", "replacement card",
            "replaced my card", "new card issued", "lost card replacement",
        ],
        "category": "Service",
        "business": "customer_support",
        "anchors": [
            "my replacement card never arrived",
            "they issued a new card very quickly",
        ],
    },
    {
        "name": "service",
        "terms": [
            "overall service", "service quality", "level of service",
            "customer care",
        ],
        "category": "Service",
        "business": "customer_support",
        "anchors": [
            "the overall service level is disappointing",
            "customer care was excellent",
        ],
    },
    {
        "name": "help documents",
        "terms": [
            "help documents", "documentation", "docs", "user guide",
            "manual", "faq", "knowledge base", "help center", "help centre",
        ],
        "category": "Support_Content",
        "business": "customer_support",
        "anchors": [
            "the documentation is outdated and confusing",
            "the FAQ answered my question immediately",
        ],
    },

    # ── Returns & refunds ────────────────────────────────────────────────────
    {
        "name": "return policy",
        "terms": [
            "return policy", "returns", "refund", "refunds", "money back",
            "refund policy", "exchange policy", "returning the product",
        ],
        "category": "Policy",
        "business": "return_and_refund",
        "anchors": [
            "the refund policy is unfair",
            "I want my money back",
            "the return policy is very clear",
        ],
    },
    {
        "name": "return process",
        "terms": [
            "return process", "returns process", "process to return",
            "how to return", "return was easy", "return was difficult",
        ],
        "category": "Policy",
        "business": "return_and_refund",
        "anchors": [
            "the return process is very complicated",
            "returning the item was quick and easy",
        ],
    },
    {
        "name": "refund speed",
        "terms": [
            "refund speed", "speed of the refund", "refund was fast",
            "refund was quick", "fast refund", "quick refund",
        ],
        "category": "Policy",
        "business": "return_and_refund",
        "anchors": [
            "the refund took three weeks to arrive",
            "I received my refund very quickly",
        ],
    },

    # ── Overall experience ───────────────────────────────────────────────────
    {
        "name": "overall experience",
        "terms": [
            "overall experience", "experience overall",
            "overall i'm happy", "overall i'm unhappy",
        ],
        "category": "Overall_Experience",
        "business": "overall_experience",
        "anchors": [
            "my overall experience has been great",
            "I am unhappy with the entire experience",
        ],
    },
]


# ═════════════════════════════════════════════════════════════════════════════
# 3.  OPINION SEED PHRASES  (replaces the bloated POS/NEG/NEU lists)
#
#  Instead of 350+ manually curated strings with duplicates, we keep ~30 clean
#  seeds per polarity.  The embedding engine generalises from these seeds to
#  any paraphrase.  The seeds still allow exact matching as a fast path.
# ═════════════════════════════════════════════════════════════════════════════

_OPINION_SEEDS: Dict[str, List[str]] = {
    "positive": [
        "fantastic", "excellent", "amazing", "great", "awesome", "wonderful",
        "superb", "brilliant", "perfect", "love it", "really like it",
        "very satisfied", "very happy", "impressed", "helpful", "friendly",
        "easy to use", "intuitive", "fast", "smooth", "reliable", "solid",
        "good value", "worth every penny", "well packaged", "arrived on time",
        "fixed it right away", "professional", "kind", "appreciate",
    ],
    "negative": [
        "terrible", "awful", "horrible", "very bad", "poor", "unacceptable",
        "disappointing", "frustrating", "hate it", "do not like", "unhappy",
        "angry", "furious", "upset", "slow", "laggy", "buggy", "crashing",
        "freezing", "broken", "not working", "damaged", "late", "delayed",
        "rude", "unhelpful", "overpriced", "too expensive", "unfair",
        "confusing", "hard to use", "difficult", "annoying", "dangerous",
        "not safe", "charged twice", "billing error", "unauthorized charge",
    ],
    "neutral": [
        "okay", "fine", "average", "decent", "acceptable", "not bad",
        "so-so", "nothing special", "has been fine",
    ],
}

# Flat lookup: phrase → polarity  (used for O(1) exact match)
_EXACT_OPINION: Dict[str, str] = {}
for _pol, _phrases in _OPINION_SEEDS.items():
    for _ph in _phrases:
        _EXACT_OPINION[_ph.lower()] = _pol


# ═════════════════════════════════════════════════════════════════════════════
# 4.  EMBEDDING LAYER  (sentence-transformers, lazy-loaded, thread-safe)
# ═════════════════════════════════════════════════════════════════════════════

_EMB_LOCK = Lock()
_EMB_MODEL = None          # SentenceTransformer instance or None if unavailable
_EMBEDDING_AVAILABLE = True  # flipped to False on first import failure

# Pre-computed anchor embeddings: {aspect_name: ndarray}
_ASPECT_ANCHOR_EMBS: Dict[str, "np.ndarray"] = {}
# Pre-computed opinion seed embeddings per polarity: {polarity: ndarray (N×D)}
_OPINION_SEED_EMBS: Dict[str, "np.ndarray"] = {}
_OPINION_SEED_LABELS: Dict[str, List[str]] = {}  # polarity → list of seeds


def _ensure_emb_model():
    """Lazy-load the embedding model once; silently disable if unavailable."""
    global _EMB_MODEL, _EMBEDDING_AVAILABLE
    if not _EMBEDDING_AVAILABLE:
        return None
    if _EMB_MODEL is not None:
        return _EMB_MODEL
    with _EMB_LOCK:
        if _EMB_MODEL is not None:
            return _EMB_MODEL
        try:
            from sentence_transformers import SentenceTransformer
            _EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            _precompute_embeddings()
            logger.info("[ABSA] Embedding model loaded: all-MiniLM-L6-v2")
        except Exception as e:
            _EMBEDDING_AVAILABLE = False
            logger.warning(
                "[ABSA] sentence-transformers not available; "
                "falling back to exact-match-only mode. Error: %s", e
            )
    return _EMB_MODEL


def _precompute_embeddings():
    """Pre-compute all anchor and seed embeddings once at load time."""
    global _ASPECT_ANCHOR_EMBS, _OPINION_SEED_EMBS, _OPINION_SEED_LABELS
    import numpy as np

    # Aspect anchors
    for spec in ASPECT_SPECS:
        anchors = spec.get("anchors", [])
        if anchors:
            embs = _EMB_MODEL.encode(anchors, convert_to_numpy=True, normalize_embeddings=True)
            _ASPECT_ANCHOR_EMBS[spec["name"]] = embs  # shape (n_anchors, D)

    # Opinion seeds per polarity
    for polarity, seeds in _OPINION_SEEDS.items():
        embs = _EMB_MODEL.encode(seeds, convert_to_numpy=True, normalize_embeddings=True)
        _OPINION_SEED_EMBS[polarity] = embs       # shape (n_seeds, D)
        _OPINION_SEED_LABELS[polarity] = seeds


def _cosine_max(vec: "np.ndarray", matrix: "np.ndarray") -> float:
    """Max cosine similarity between a single vector and each row in matrix."""
    # Both are L2-normalised, so cosine = dot product
    scores = matrix @ vec
    return float(scores.max())


# ═════════════════════════════════════════════════════════════════════════════
# 5.  NEGATION DETECTION
# ═════════════════════════════════════════════════════════════════════════════

_NEGATION_TOKENS = {
    "not", "n't", "never", "no", "neither", "nor", "without",
    "hardly", "barely", "scarcely",
}
_NEGATION_WINDOW = 4   # tokens to the left of the opinion phrase


def _is_negated(text_tokens: List[str], opinion_start_token: int) -> bool:
    """Return True if a negation word appears within the window before the opinion."""
    window_start = max(0, opinion_start_token - _NEGATION_WINDOW)
    window = text_tokens[window_start:opinion_start_token]
    return any(t.lower() in _NEGATION_TOKENS for t in window)


def _flip_polarity(sentiment: str) -> str:
    return {"positive": "negative", "negative": "positive"}.get(sentiment, sentiment)


# ═════════════════════════════════════════════════════════════════════════════
# 6.  CONFIDENCE FORMULA
# ═════════════════════════════════════════════════════════════════════════════

def _confidence(
    match_type: str,         # "exact" | "embedding" | "global_fallback"
    similarity: float,       # embedding similarity score (0–1), or 1.0 for exact
    distance: int,           # char distance between aspect and opinion in text
    emotion_polarity: Optional[str],   # from emotion pipeline
    opinion_sentiment: str,  # derived sentiment
) -> float:
    """
    Confidence = base_weight × distance_decay × alignment_bonus

    base_weight:
        exact           → 0.95
        embedding ≥0.75 → 0.85
        embedding ≥0.60 → 0.72
        global_fallback → 0.55

    distance_decay:
        exp(-distance / 120)  — soft exponential; same-sentence (~50 chars) ≈ 0.67

    alignment_bonus:
        +0.05 if opinion sentiment matches emotion polarity (capped at 1.0)
    """
    if match_type == "exact":
        base = 0.95
    elif similarity >= 0.75:
        base = 0.85
    elif similarity >= 0.60:
        base = 0.72
    else:
        base = 0.55  # global_fallback or weak embedding

    decay = math.exp(-distance / 120.0)
    conf = base * decay

    # Alignment bonus: emotion polarity agrees with derived opinion sentiment
    if emotion_polarity and emotion_polarity in {"positive", "negative", "neutral"}:
        if opinion_sentiment == emotion_polarity:
            conf = min(1.0, conf + 0.05)

    return round(conf, 3)


# ═════════════════════════════════════════════════════════════════════════════
# 7.  EXACT MATCHERS  (same logic as before, cleaned up)
# ═════════════════════════════════════════════════════════════════════════════

def _word_boundary_match(text_lc: str, term_lc: str, start: int) -> Optional[Tuple[int, int]]:
    """
    Find the next occurrence of term_lc in text_lc starting at `start`,
    enforcing that the match is at a word boundary (not inside another word).
    Returns (idx, end) or None.
    """
    pos = start
    while True:
        idx = text_lc.find(term_lc, pos)
        if idx == -1:
            return None
        end = idx + len(term_lc)
        # Check left boundary: must be start-of-string or a non-alphanumeric char
        left_ok = (idx == 0) or (not text_lc[idx - 1].isalnum() and text_lc[idx - 1] != "'")
        # Check right boundary: must be end-of-string or a non-alphanumeric char
        right_ok = (end == len(text_lc)) or (not text_lc[end].isalnum() and text_lc[end] != "'")
        if left_ok and right_ok:
            return idx, end
        pos = idx + 1
    return None


def _find_aspect_mentions_exact(text: str) -> List[Dict]:
    """Return all non-overlapping aspect mentions via exact substring match.
    Word-boundary enforcement prevents 'app' matching inside 'happy'."""
    text_lc = text.lower()
    mentions: List[Dict] = []
    used: List[Tuple[int, int]] = []

    for spec in ASPECT_SPECS:
        for term in sorted(spec["terms"], key=len, reverse=True):
            term_lc = term.lower()
            search_from = 0
            while True:
                result = _word_boundary_match(text_lc, term_lc, search_from)
                if result is None:
                    break
                idx, end = result
                if any(not (end <= s or idx >= e) for s, e in used):
                    search_from = idx + 1
                    continue
                mentions.append({
                    "aspect":      spec["name"],
                    "business":    spec["business"],
                    "category":    spec["category"],
                    "aspect_term": text[idx:end],
                    "match_type":  "exact",
                    "similarity":  1.0,
                    "pos":         idx,
                })
                used.append((idx, end))
                search_from = idx + 1

    return mentions


def _find_opinion_mentions_exact(text: str) -> List[Dict]:
    """Return all non-overlapping opinion mentions via exact substring match,
    with word-boundary enforcement."""
    text_lc = text.lower()
    tokens = re.findall(r"\w+|\S", text_lc)

    mentions: List[Dict] = []
    used: List[Tuple[int, int]] = []

    for phrase in sorted(_EXACT_OPINION.keys(), key=len, reverse=True):
        search_from = 0
        while True:
            result = _word_boundary_match(text_lc, phrase, search_from)
            if result is None:
                break
            idx, end = result
            if any(not (end <= s or idx >= e) for s, e in used):
                search_from = idx + 1
                continue

            # Negation check
            char_before = text_lc[:idx]
            tok_idx = len(re.findall(r"\w+|\S", char_before))
            negated = _is_negated(tokens, tok_idx)

            raw_sentiment = _EXACT_OPINION[phrase]
            sentiment = _flip_polarity(raw_sentiment) if negated else raw_sentiment

            mentions.append({
                "opinion_term": text[idx:end],
                "sentiment":    sentiment,
                "negated":      negated,
                "match_type":   "exact",
                "similarity":   1.0,
                "pos":          idx,
            })
            used.append((idx, end))
            search_from = idx + 1

    return mentions


# ═════════════════════════════════════════════════════════════════════════════
# 8.  EMBEDDING MATCHERS  (only called when exact matching finds nothing)
# ═════════════════════════════════════════════════════════════════════════════

_EMB_ASPECT_THRESHOLD = 0.55   # below this → ignore
_EMB_OPINION_THRESHOLD = 0.52


def _find_aspects_by_embedding(text: str) -> List[Dict]:
    """
    Embed the full text and compare to each aspect's anchor embeddings.
    Returns aspects ranked by max similarity, deduplicated.
    Only called when exact match finds nothing.
    """
    model = _ensure_emb_model()
    if model is None or not _ASPECT_ANCHOR_EMBS:
        return []

    import numpy as np

    text_vec = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

    results: List[Dict] = []
    seen_aspects: set = set()

    scored = []
    for spec in ASPECT_SPECS:
        anchor_embs = _ASPECT_ANCHOR_EMBS.get(spec["name"])
        if anchor_embs is None:
            continue
        sim = _cosine_max(text_vec, anchor_embs)
        if sim >= _EMB_ASPECT_THRESHOLD:
            scored.append((sim, spec))

    # Keep top-3 by similarity to avoid noise
    scored.sort(key=lambda x: x[0], reverse=True)
    for sim, spec in scored[:3]:
        if spec["name"] in seen_aspects:
            continue
        seen_aspects.add(spec["name"])
        results.append({
            "aspect":      spec["name"],
            "business":    spec["business"],
            "category":    spec["category"],
            "aspect_term": None,   # embedding match — no surface term
            "match_type":  "embedding",
            "similarity":  round(float(sim), 3),
            "pos":         0,      # whole-text match; no character position
        })

    return results


def _find_opinion_by_embedding(text: str) -> Optional[Dict]:
    """
    Embed the full text and score it against positive/negative/neutral seed pools.
    Returns the winning polarity or None if below threshold.
    Only called when exact opinion matching finds nothing.
    """
    model = _ensure_emb_model()
    if model is None or not _OPINION_SEED_EMBS:
        return None

    import numpy as np

    text_vec = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

    best_pol: Optional[str] = None
    best_sim = _EMB_OPINION_THRESHOLD  # must beat this

    for polarity, seed_embs in _OPINION_SEED_EMBS.items():
        sim = _cosine_max(text_vec, seed_embs)
        if sim > best_sim:
            best_sim = sim
            best_pol = polarity

    if best_pol is None:
        return None

    return {
        "opinion_term": None,
        "sentiment":    best_pol,
        "negated":      False,
        "match_type":   "embedding",
        "similarity":   round(best_sim, 3),
        "pos":          0,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 9.  CONTRAST CLAUSE DETECTION  (unchanged logic, kept for mixed messages)
# ═════════════════════════════════════════════════════════════════════════════

_CONTRAST_MARKERS = [" but ", " even though ", " although ", " though ", " however "]
_NEG_CONTEXT_HINTS = [
    "too high", "unjustifiable", "outrageous", "ridiculous",
    "incorrect", "wrong", "broken", "not working", "doesn't work", "doesnt work",
    "slow", "lags", "laggy", "freezes", "crashes", "buggy",
    "unhelpful", "rude", "late", "delayed",
]


def _detect_contrast_clause(text: str) -> Tuple[bool, Optional[int]]:
    """
    Returns (second_clause_is_negative, second_clause_start_char_idx).
    """
    text_lc = text.lower()
    for marker in _CONTRAST_MARKERS:
        idx = text_lc.find(marker)
        if idx != -1:
            after = text_lc[idx + len(marker):]
            neg = any(h in after for h in _NEG_CONTEXT_HINTS)
            return neg, idx + len(marker)
    return False, None


# ═════════════════════════════════════════════════════════════════════════════
# 10. ASPECT–OPINION PAIRING
# ═════════════════════════════════════════════════════════════════════════════

def _pair_aspects_opinions(
    aspect_mentions: List[Dict],
    opinion_mentions: List[Dict],
    text: str,
    global_sentiment: Optional[str],
    emotion_polarity: Optional[str],
    second_clause_negative: bool,
    second_clause_start: Optional[int],
) -> List[Dict]:
    """
    Greedy nearest-neighbour pairing.  Same-sentence aspects get a proximity
    bonus.  Unpaired aspects fall back to global_sentiment.
    """
    if not opinion_mentions:
        # No opinions at all → use global sentiment for all aspects
        results = []
        for a in aspect_mentions:
            sent = global_sentiment or "neutral"
            conf_base = 0.7 if global_sentiment else 0.5

            # Contrast-clause override
            if second_clause_negative and second_clause_start is not None:
                if a.get("pos", 0) >= second_clause_start:
                    sent = "negative"
                    conf_base = 0.85

            conf = _confidence(
                match_type=a["match_type"],
                similarity=a.get("similarity", 1.0),
                distance=0,
                emotion_polarity=emotion_polarity,
                opinion_sentiment=sent,
            )
            results.append(_make_result(a, None, sent, conf))
        return results

    # Build (distance, aspect_idx, opinion_idx) triples
    pairs = []
    for ai, a in enumerate(aspect_mentions):
        for oi, o in enumerate(opinion_mentions):
            # Same-sentence bonus: look for shared period boundaries
            a_sent_start = text.rfind(".", 0, a["pos"])
            a_sent_end   = text.find(".", a["pos"])
            o_sent_start = text.rfind(".", 0, o["pos"])

            same_sentence = (a_sent_start == o_sent_start)
            dist = abs(a["pos"] - o["pos"])
            if same_sentence:
                dist = max(0, dist - 30)

            pairs.append((dist, ai, oi))

    pairs.sort(key=lambda x: x[0])

    assigned_aspects:  set = set()
    assigned_opinions: set = set()
    results: List[Dict] = []

    for dist, ai, oi in pairs:
        if ai in assigned_aspects or oi in assigned_opinions:
            continue
        assigned_aspects.add(ai)
        assigned_opinions.add(oi)

        a = aspect_mentions[ai]
        o = opinion_mentions[oi]
        conf = _confidence(
            match_type=a["match_type"],
            similarity=a.get("similarity", 1.0),
            distance=dist,
            emotion_polarity=emotion_polarity,
            opinion_sentiment=o["sentiment"],
        )
        results.append(_make_result(a, o["opinion_term"], o["sentiment"], conf))

    # Unpaired aspects
    for ai, a in enumerate(aspect_mentions):
        if ai in assigned_aspects:
            continue
        sent = global_sentiment or "neutral"
        if second_clause_negative and second_clause_start is not None:
            if a.get("pos", 0) >= second_clause_start:
                sent = "negative"

        conf = _confidence(
            match_type=a["match_type"],
            similarity=a.get("similarity", 1.0),
            distance=0,
            emotion_polarity=emotion_polarity,
            opinion_sentiment=sent,
        )
        results.append(_make_result(a, None, sent, conf))

    return results


def _make_result(
    aspect: Dict,
    opinion_term: Optional[str],
    sentiment: str,
    confidence: float,
) -> Dict:
    """Build the canonical output dict consumed by actions.py and the CSV layer."""
    return {
        "aspect":                      aspect["aspect"],
        "aspect_term":                 aspect.get("aspect_term"),
        "aspect_category":             aspect["category"],
        "business_aspect":             aspect["business"],
        "business_aspect_display":     BUSINESS_ASPECTS.get(aspect["business"], aspect["business"]),
        "opinion_term":                opinion_term,
        "aspect_sentiment":            sentiment,
        "aspect_sentiment_confidence": confidence,
        "aspect_match_type":           aspect["match_type"],
        "aspect_similarity":           aspect.get("similarity", 1.0),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 11. PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

def extract_aspect_opinions(
    text: str,
    sentiment_label: Optional[str] = None,
    emotion_label: Optional[str] = None,
) -> List[Dict]:
    """
    Main ABSA entry point called by actions.py.

    Returns a list of dicts, one per detected aspect:
      {
        aspect                      : str  – canonical fine-grained name
        aspect_term                 : str | None  – surface phrase in text
        aspect_category             : str  – technical grouping
        business_aspect             : str  – one of 7 business aspect keys
        business_aspect_display     : str  – human-readable label for dashboard
        opinion_term                : str | None
        aspect_sentiment            : "positive" | "negative" | "neutral"
        aspect_sentiment_confidence : float  (0–1)
        aspect_match_type           : "exact" | "embedding" | "global_fallback"
        aspect_similarity           : float  (1.0 for exact, 0–1 for embedding)
      }
    """
    original_text = (text or "").strip()
    if not original_text:
        return []
    # Skip Rasa payload commands like "/cancel_order"
    if original_text.startswith("/"):
        return []

    # ── Resolve global sentiment prior ───────────────────────────────────────
    global_sentiment: Optional[str] = None
    emotion_polarity: Optional[str] = None

    if sentiment_label in {"positive", "negative", "neutral"}:
        global_sentiment = sentiment_label
    elif emotion_label:
        coarse = coarse_from_label(emotion_label)
        if coarse in {"positive", "negative", "neutral"}:
            global_sentiment = coarse

    if emotion_label:
        p = coarse_from_label(emotion_label)
        if p in {"positive", "negative", "neutral"}:
            emotion_polarity = p

    # ── Step 1: Exact aspect matching ─────────────────────────────────────────
    aspect_mentions = _find_aspect_mentions_exact(original_text)

    # ── Step 2: Embedding fallback (if exact found nothing) ───────────────────
    used_embedding_fallback = False
    if not aspect_mentions:
        aspect_mentions = _find_aspects_by_embedding(original_text)
        used_embedding_fallback = bool(aspect_mentions)

    # ── Still nothing → overall experience synthetic aspect ───────────────────
    if not aspect_mentions:
        if global_sentiment is not None:
            # Require meaningful text (>5 tokens) before creating synthetic aspect
            if len(original_text.split()) >= 5:
                return [{
                    "aspect":                      "overall experience",
                    "aspect_term":                 None,
                    "aspect_category":             "Overall_Experience",
                    "business_aspect":             "overall_experience",
                    "business_aspect_display":     BUSINESS_ASPECTS["overall_experience"],
                    "opinion_term":                None,
                    "aspect_sentiment":            global_sentiment,
                    "aspect_sentiment_confidence": 0.55,
                    "aspect_match_type":           "global_fallback",
                    "aspect_similarity":           0.0,
                }]
        return []

    # ── Step 3: Exact opinion matching ────────────────────────────────────────
    opinion_mentions = _find_opinion_mentions_exact(original_text)

    # ── Step 4: Embedding opinion fallback (if exact found nothing) ───────────
    if not opinion_mentions:
        emb_opinion = _find_opinion_by_embedding(original_text)
        if emb_opinion:
            # Treat as a single whole-text opinion signal
            opinion_mentions = [emb_opinion]

    # ── Step 5: Contrast clause detection ─────────────────────────────────────
    second_clause_neg, second_clause_start = _detect_contrast_clause(original_text)

    # ── Step 6: Pair aspects ↔ opinions → build result table ─────────────────
    results = _pair_aspects_opinions(
        aspect_mentions=aspect_mentions,
        opinion_mentions=opinion_mentions,
        text=original_text,
        global_sentiment=global_sentiment,
        emotion_polarity=emotion_polarity,
        second_clause_negative=second_clause_neg,
        second_clause_start=second_clause_start,
    )

    logger.debug(
        "[ABSA-v2] text=%r | aspects=%d (emb=%s) | opinions=%d | results=%d",
        original_text[:80],
        len(aspect_mentions),
        used_embedding_fallback,
        len(opinion_mentions),
        len(results),
    )

    return results