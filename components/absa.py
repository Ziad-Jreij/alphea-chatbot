# components/absa.py

from __future__ import annotations
from typing import List, Dict, Optional
from components.emotion import coarse_from_label  # reuse if we need global sentiment fallback

# ---- Aspect taxonomy for generic product feedback ----
# name: what you store in "Aspect"
# terms: how it might appear in text
# category: high-level category for thesis (Performance, Features, Cost, Aesthetics, etc.)

ASPECT_SPECS = [
    # --- Product performance & features ---
    {
        "name": "battery life",
        "terms": ["battery life", "battery"],
        "category": "Performance",
    },
    {
        "name": "camera",
        "terms": ["camera", "photos", "pictures", "photo quality"],
        "category": "Features",
    },
    {
        "name": "product quality",
        "terms": ["quality", "build quality", "materials"],
        "category": "Features",
    },
    {
        "name": "performance",
        "terms": ["performance", "speed", "lag", "lagging", "slow", "snappy"],
        "category": "Performance",
    },
    {
        "name": "reliability",
        "terms": ["reliability", "reliable", "crashes", "crashing", "bugs", "buggy"],
        "category": "Reliability",
    },
    {
        "name": "usability",
        "terms": ["usability", "ease of use", "easy to use", "user friendly", "user-friendly", "intuitive"],
        "category": "Usability",
    },

    # --- Design / aesthetics ---
    {
        "name": "design",
        "terms": ["design", "look", "appearance", "style"],
        "category": "Aesthetics",
    },

    # --- Price / value ---
    {
        "name": "price",
        "terms": ["price", "cost", "pricing"],
        "category": "Cost",
    },
    {
        "name": "value for money",
        "terms": ["value for money", "worth it", "good value"],
        "category": "Cost",
    },

    # --- Delivery / logistics ---
    {
        "name": "delivery",
        "terms": ["delivery", "shipping"],
        "category": "Logistics",
    },
    {
        "name": "delivery time",
        "terms": ["delivery time", "shipping time", "arrived on time", "arrived late"],
        "category": "Logistics",
    },
    {
        "name": "packaging",
        "terms": ["packaging", "package", "box"],
        "category": "Logistics",
    },

    # --- Support / service ---
    {
        "name": "customer support",
        "terms": ["support", "customer support", "customer service", "helpdesk"],
        "category": "Service",
    },
    {
        "name": "response time",
        "terms": ["response time", "reply time", "responded quickly", "slow to respond"],
        "category": "Service",
    },

    # --- Digital experience ---
    {
        "name": "app experience",
        "terms": ["app", "mobile app", "application"],
        "category": "Digital_Experience",
    },
    {
        "name": "website experience",
        "terms": ["website", "site", "web page"],
        "category": "Digital_Experience",
    },

    # --- Policies / returns ---
    {
        "name": "return policy",
        "terms": ["return policy", "returns", "refund", "refunds"],
        "category": "Policy",
    },

    # --- Overall experience ---
    {
        "name": "overall experience",
        "terms": ["experience overall", "overall experience", "overall"],
        "category": "Overall_Experience",
    },
]


# ---- Opinion lexicon (very small but explainable) ----

POS_OPINIONS = [
    # General praise
    "fantastic", "great", "amazing", "excellent", "awesome",
    "perfect", "wonderful", "superb", "brilliant",

    # Satisfaction / liking
    "love", "like", "really like", "quite like",
    "satisfied", "very satisfied", "happy", "very happy",
    "impressed", "pleased",

    # Quality / performance
    "high quality", "good quality", "solid", "reliable", "stable",
    "smooth", "snappy", "fast", "responsive",

    # Usability
    "easy to use", "user friendly", "user-friendly", "intuitive",
    "simple", "straightforward",

    # Service / delivery
    "helpful", "very helpful", "friendly",
    "on time", "arrived on time", "well packaged",
]


NEG_OPINIONS = [
    # Strong negative
    "bad", "very bad", "terrible", "awful", "horrible", "poor",
    "horrendous", "unacceptable",

    # Dissatisfaction / dislike
    "disappointing", "very disappointing",
    "hate", "really hate", "do not like", "don't like",

    # Price / cost
    "too high", "very high", "overpriced", "expensive", "way too expensive",

    # Quality / performance / reliability
    "low quality", "cheap", "flimsy",
    "slow", "very slow", "laggy", "lagging",
    "buggy", "crashes", "keeps crashing", "unstable",

    # Usability / UX
    "hard to use", "difficult to use", "confusing", "not intuitive",
    "complicated", "clunky",

    # Service / delivery
    "unhelpful", "not helpful", "unfriendly", "rude",
    "late", "very late", "delayed", "arrived damaged", "damaged",
]


NEU_OPINIONS = [
    "average", "just average", "okay", "ok", "fine",
    "so so", "so-so", "decent", "acceptable",
    "not bad", "nothing special",
]


def _find_phrases_in_window(
    window_text_lc: str,
    window_offset: int,
    center_idx: int,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Search for opinion phrases in the window, return:
      (opinion_term, sentiment_label, confidence_tag)
    sentiment_label in {'positive','negative','neutral'} or None
    confidence_tag in {'high','medium','low'} or None
    """
    def closest_match(phrases: List[str], sentiment: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        best_term = None
        best_dist = None
        for phrase in phrases:
            i = window_text_lc.find(phrase)
            if i == -1:
                continue
            global_pos = window_offset + i
            dist = abs(global_pos - center_idx)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_term = phrase
        if best_term is None:
            return None, None, None
        # high confidence if we found exact lexicon phrase
        return best_term, sentiment, "high"

    # Try positive, negative, neutral in that order
    term, sent, conf = closest_match(POS_OPINIONS, "positive")
    if term:
        return term, sent, conf

    term, sent, conf = closest_match(NEG_OPINIONS, "negative")
    if term:
        return term, sent, conf

    term, sent, conf = closest_match(NEU_OPINIONS, "neutral")
    if term:
        return term, sent, conf

    return None, None, None


def _confidence_from_tag(tag: Optional[str]) -> float:
    if tag == "high":
        return 0.95
    if tag == "medium":
        return 0.85
    if tag == "low":
        return 0.7
    return 0.5


def extract_aspect_opinions(
    text: str,
    sentiment_label: Optional[str] = None,
    emotion_label: Optional[str] = None,
) -> List[Dict]:
    """
    Heuristic ABSA extractor:
      - finds aspect terms from ASPECT_SPECS
      - finds closest opinion term near each aspect
      - assigns aspect-level sentiment from the opinion term (or global sentiment fallback)

    Returns list of dicts:
      {
        "aspect": "battery life",
        "aspect_category": "Performance",
        "opinion_term": "fantastic",
        "aspect_sentiment": "positive",
        "aspect_sentiment_confidence": 0.95,
      }
    """
    original_text = (text or "").strip()
    if not original_text:
        return []

    text_lc = original_text.lower()
    results: List[Dict] = []

    # Optional global fallback sentiment if no opinion term found
    global_sentiment = None
    if sentiment_label in {"positive", "negative", "neutral"}:
        global_sentiment = sentiment_label
    elif emotion_label:
        coarse = coarse_from_label(emotion_label)
        if coarse in {"positive", "negative", "neutral"}:
            global_sentiment = coarse

    for spec in ASPECT_SPECS:
        aspect_name = spec["name"]
        category = spec["category"]

        found_any_for_this_aspect = False

        for term in spec["terms"]:
            term_lc = term.lower()
            start = text_lc.find(term_lc)
            if start == -1:
                continue

            found_any_for_this_aspect = True
            end = start + len(term_lc)

            # Expand window around the aspect term to search for opinion words
            window_radius = 60  # characters after the aspect term
            win_start = end
            win_end = min(len(text_lc), end + window_radius)
            window_text_lc = text_lc[win_start:win_end]

            opinion_term_lc, sent, conf_tag = _find_phrases_in_window(
                window_text_lc=window_text_lc,
                window_offset=win_start,
                center_idx=end,
            )

            # Use original casing for opinion term
            opinion_term = None
            if opinion_term_lc:
                i = text_lc.find(opinion_term_lc, win_start, win_end)
                if i != -1:
                    opinion_term = original_text[i : i + len(opinion_term_lc)]

            aspect_sentiment = sent
            if not aspect_sentiment:
                # Fallback to global sentiment if we have it
                if global_sentiment:
                    aspect_sentiment = global_sentiment
                    conf_tag = "medium"
                else:
                    aspect_sentiment = "neutral"
                    conf_tag = "low"

            conf_score = _confidence_from_tag(conf_tag)

            # Use the surface aspect phrase from the original text
            aspect_term = original_text[start:end]

            results.append(
                {
                    "aspect": aspect_name,          # canonical name, e.g. "battery life"
                    "aspect_term": aspect_term,     # surface phrase, e.g. "battery life"
                    "aspect_category": category,    # e.g. "Performance"
                    "opinion_term": opinion_term,   # e.g. "fantastic"
                    "aspect_sentiment": aspect_sentiment,
                    "aspect_sentiment_confidence": conf_score,
                }
            )

            # For now, one occurrence per aspect is enough
            break

        # If you ever want multiple occurrences per aspect in long texts,
        # you can remove the 'break' and loop via text_lc.find in a while.
    return results
