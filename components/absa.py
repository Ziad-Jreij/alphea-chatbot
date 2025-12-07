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
        "terms": [
            "battery life",
            "battery",
            "battery duration",
            "battery performance",
            "battery power",
            "battery drains",
            "battery drain",
            "battery dies fast",
            "battery keeps dying",
            "charge lasts",
            "one charge",
            "holds a charge",
        ],
        "category": "Performance",
    },
    {
        "name": "camera",
        "terms": [
            "camera",
            "photo quality",
            "photos",
            "pictures",
            "pics",
            "image quality",
            "takes great photos",
            "takes bad photos",
            "selfie camera",
            "front camera",
            "rear camera",
        ],
        "category": "Features",
    },
    {
        "name": "product quality",
        "terms": [
            "product quality",
            "quality",
            "overall quality",
            "build quality",
            "build",
            "durability",
            "durable",
            "sturdy",
            "well built",
            "poorly built",
            "cheap materials",
            "materials",
            "material quality",
            "feels cheap",
            "feels solid",
        ],
        "category": "Features",
    },
    {
        "name": "performance",
        "terms": [
            "performance",
            "speed",
            "fast",
            "snappy",
            "smooth",
            "slow",
            "lag",
            "laggy",
            "lagging",
            "freezes",
            "keeps freezing",
            "stutters",
            "loading time",
            "load time",
            "runs well",
            "runs poorly",
            "responsive",
            "responsiveness",
        ],
        "category": "Performance",
    },
    {
        "name": "reliability",
        "terms": [
            "reliability",
            "reliable",
            "unreliable",
            "crashes",
            "crashing",
            "keeps crashing",
            "crashed",
            "bugs",
            "buggy",
            "glitches",
            "glitchy",
            "frozen",
            "keeps restarting",
            "hangs",
            "unstable",
            "stability",
        ],
        "category": "Reliability",
    },
    {
        "name": "internet speed",
        "terms": [
            "internet speed",
            "speed of my internet",
            "slow internet",
            "connection speed",
            "bandwidth",
            "slow connection",
            "fast connection",
            "download speed",
            "upload speed",
        ],
        "category": "Performance",
    },
    {
        "name": "software update",
        "terms": [
            "software update",
            "app update",
            "system update",
            "latest version",
            "new version",
            "update broke",
            "after the update",
        ],
        "category": "Features",
    },
    {
        "name": "main feature",
        "terms": [
            "main feature",
            "key feature",
            "core feature",
            "primary feature",
            "the feature i needed",
            "feature i wanted",
        ],
        "category": "Features",
    },

    # --- Usability / ease of use ---
    {
        "name": "usability",
        "terms": [
            "usability",
            "ease of use",
            "easy to use",
            "simple to use",
            "user friendly",
            "user-friendly",
            "intuitive",
            "not intuitive",
            "hard to use",
            "difficult to use",
            "complicated to use",
            "confusing to use",
            "navigation",
            "hard to navigate",
            "easy to navigate",
            "interface is confusing",
            "UI is confusing",
        ],
        "category": "Usability",
    },
    {
        "name": "size guide",
        "terms": [
            "size guide",
            "sizing guide",
            "size chart",
            "sizing chart",
            "fit guide",
            "didn't match the size chart",
            "sizes are off",
        ],
        "category": "Usability",
    },

    # --- Design / aesthetics ---
    {
        "name": "design",
        "terms": [
            "design",
            "look",
            "looks",
            "appearance",
            "style",
            "aesthetic",
            "aesthetics",
            "visual design",
            "color",
            "colour",
            "shape",
            "sleek",
            "modern look",
            "ugly",
        ],
        "category": "Aesthetics",
    },

    # --- Price / value / billing ---
    {
        "name": "price",
        "terms": [
            "price",
            "prices",
            "cost",
            "expensive",
            "too expensive",
            "overpriced",
            "cheap",
            "affordable",
            "pricing",
            "price point",
            "price tag",
        ],
        "category": "Cost",
    },
    {
        "name": "value for money",
        "terms": [
            "value for money",
            "good value",
            "great value",
            "poor value",
            "not worth it",
            "worth it",
            "worth the money",
            "worth the price",
            "worth every penny",
            "bang for the buck",
        ],
        "category": "Cost",
    },
    {
        "name": "billing",
        "terms": [
            "billing",
            "bill",
            "bills",
            "charge",
            "charges",
            "extra charge",
            "charged twice",
            "wrong charge",
            "charge on my statement",
            "statement error",
            "billing issue",
            "billing problems",
        ],
        "category": "Cost",
    },
    {
        "name": "fees",
        "terms": [
            "fees",
            "fee",
            "atm fees",
            "hidden fees",
            "extra fees",
            "service fee",
            "maintenance fee",
            "late fee",
        ],
        "category": "Cost",
    },
    {
        "name": "loan terms",
        "terms": [
            "loan terms",
            "loan conditions",
            "interest rate",
            "interest rates",
            "loan agreement",
            "repayment terms",
            "installments",
        ],
        "category": "Cost",
    },

    # --- Delivery / logistics ---
    {
        "name": "delivery & shipping",
        "terms": [
            "delivery",
            "shipping",
            "shipped",
            "delivered",
            "shipping process",
            "delivery service",
            "courier",
            "package delivery",
            "shipping company",
        ],
        "category": "Logistics",
    },
    {
        "name": "delivery time",
        "terms": [
            "delivery time",
            "shipping time",
            "time to deliver",
            "arrived on time",
            "arrived late",
            "came late",
            "took forever to arrive",
            "shipping took forever",
            "shipping took long",
            "delayed delivery",
            "delivery delay",
            "fast delivery",
            "quick delivery",
        ],
        "category": "Logistics",
    },
    {
        "name": "packaging",
        "terms": [
            "packaging",
            "package",
            "the box",
            "box was damaged",
            "well packaged",
            "poorly packaged",
            "damaged packaging",
            "unboxing",
        ],
        "category": "Logistics",
    },

    # --- Digital experience (app / web / online) ---
    {
        "name": "app experience",
        "terms": [
            "mobile app",
            "phone app",
            "the app experience",
            "in the app",
            "app interface",
            "app design",
            "app layout",
        ],
        "category": "Digital_Experience",
    },
    {
        "name": "website experience",
        "terms": [
            "website",
            "site",
            "web site",
            "webpage",
            "web page",
            "on your site",
            "on the website",
            "online portal",
        ],
        "category": "Digital_Experience",
    },
    {
        "name": "order page",
        "terms": [
            "order page",
            "checkout page",
            "checkout",
            "payment page",
            "cart page",
            "shopping cart",
            "couldn't check out",
        ],
        "category": "Digital_Experience",
    },
    {
        "name": "online application process",
        "terms": [
            "online application process",
            "online application",
            "applied online",
            "application form online",
            "online form",
        ],
        "category": "Digital_Experience",
    },

    # --- Security / login / fraud ---
    {
        "name": "login",
        "terms": [
            "login",
            "log in",
            "sign in",
            "sign-in",
            "log-in",
            "logged in",
            "can't log in",
            "login issues",
        ],
        "category": "Security",
    },
    {
        "name": "two-factor authentication",
        "terms": [
            "two-factor authentication",
            "two factor authentication",
            "2fa",
            "two-step verification",
            "2-step verification",
            "verification code",
            "authenticator app",
        ],
        "category": "Security",
    },
    {
        "name": "fraud alert system",
        "terms": [
            "fraud alert system",
            "fraud alerts",
            "fraud alert",
            "suspicious activity alerts",
            "fraud detection",
            "blocked my card for fraud",
        ],
        "category": "Security",
    },

    # --- Support / service / content ---
    {
        "name": "customer support",
        "terms": [
            "customer support",
            "support team",
            "support",
            "customer service",
            "service agent",
            "agent on chat",
            "helpdesk",
            "help desk",
            "call center",
            "call centre",
            "hotline",
        ],
        "category": "Service",
    },
    {
        "name": "branch staff",
        "terms": [
            "branch staff",
            "staff at the branch",
            "staff in the branch",
            "people at the branch",
            "bank staff",
            "teller",
            "tellers",
        ],
        "category": "Service",
    },
    {
        "name": "response time",
        "terms": [
            "response time",
            "reply time",
            "responded quickly",
            "responded fast",
            "slow to respond",
            "took too long to respond",
            "quick response",
            "fast response",
            "no one replied",
        ],
        "category": "Service",
    },
    {
        "name": "card replacement",
        "terms": [
            "card replacement",
            "card was replaced",
            "replacement card",
            "replaced my card",
            "new card issued",
            "lost card replacement",
        ],
        "category": "Service",
    },
    {
        "name": "service",
        "terms": [
            "overall service",
            "service quality",
            "level of service",
            "customer care",
        ],
        "category": "Service",
    },
    {
        "name": "help documents",
        "terms": [
            "help documents",
            "documentation",
            "docs",
            "user guide",
            "manual",
            "faq",
            "knowledge base",
            "help center",
            "help centre",
        ],
        "category": "Support_Content",
    },

    # --- Policies / returns ---
    {
        "name": "return policy",
        "terms": [
            "return policy",
            "returns",
            "refund",
            "refunds",
            "money back",
            "refund policy",
            "exchange policy",
            "returning the product",
        ],
        "category": "Policy",
    },
    {
        "name": "return process",
        "terms": [
            "return process",
            "returns process",
            "process to return",
            "how to return",
            "return was easy",
            "return was difficult",
        ],
        "category": "Policy",
    },

    # --- Overall experience ---
    {
        "name": "overall experience",
        "terms": [
            "overall experience",
            "experience overall",
            "overall i'm happy",
            "overall i'm unhappy",
        ],
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
    "simple", "straightforward", "feels solid",
    "return was easy",
    "overall i'm happy",
    "worth every penny",
    "good value",
    "great value",

    # Service / delivery
    "helpful", "very helpful", "friendly",
    "on time", "arrived on time", "well packaged",
    "absolutely love", "love", "really love",
    "quick and easy", "quick", "easy",
    "sleek", "polished",
    "kind", "professional",
    "appreciate",
    "fixed it right away", "replaced quickly",
]


NEG_OPINIONS = [
    # Strong negative
    "bad", "very bad", "terrible", "awful", "horrible", "poor",
    "horrendous", "unacceptable", "feels cheap",
    "poor value",
    "not worth it",
    "return was difficult",
    "overall i'm unhappy",

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
    "took forever", "took too long",
    "frustrating", "honestly frustrating",
    "kept crashing", "crashing",
    "annoying error", "annoying error code", "annoying",
    "too high", "getting ridiculous", "ridiculous",
    "hate", "really hate",
    "incorrect", "still see an old, incorrect charge",
    
]


NEU_OPINIONS = [
    "average", "just average", "okay", "ok", "fine",
    "so so", "so-so", "decent", "acceptable",
    "not bad", "nothing special", "has been fine", "fine",
]


def _find_all_opinion_mentions(
    text: str,
    original_text: str,
) -> List[Dict]:
    """
    Find all opinion phrases (positive/negative/neutral) in the text.
    Avoid overlapping spans (e.g. 'unhelpful' vs 'helpful').
    """
    text_lc = text.lower()
    mentions: List[Dict] = []
    used_spans: List[tuple[int, int]] = []  # list of (start, end)

    def add_mentions(phrases: List[str], sentiment: str):
        nonlocal mentions, used_spans
        for phrase in phrases:
            phr = phrase.lower()
            start = 0
            while True:
                idx = text_lc.find(phr, start)
                if idx == -1:
                    break
                end = idx + len(phr)

                # Skip if this span overlaps a previously accepted opinion
                overlap = any(not (end <= s or idx >= e) for (s, e) in used_spans)
                if overlap:
                    start = idx + 1
                    continue

                surface = original_text[idx:end]
                mentions.append(
                    {
                        "opinion_term": surface,
                        "sentiment": sentiment,
                        "pos": idx,
                    }
                )
                used_spans.append((idx, end))
                start = idx + 1

    # Order matters: negatives first, then positives, then neutral
    add_mentions(NEG_OPINIONS, "negative")
    add_mentions(POS_OPINIONS, "positive")
    add_mentions(NEU_OPINIONS, "neutral")

    return mentions


def _find_all_aspect_mentions(
    text: str,
    original_text: str,
) -> List[Dict]:
    """
    Find all aspect mentions in the text based on ASPECT_SPECS.
    Avoid overlapping spans so that synonyms don't generate duplicates.
    """
    text_lc = text.lower()
    aspects: List[Dict] = []
    used_spans: List[tuple[int, int]] = []  # list of (start, end)

    for spec in ASPECT_SPECS:
        aspect_name = spec["name"]
        category = spec["category"]

        for term in spec["terms"]:
            term_lc = term.lower()
            start = 0
            while True:
                idx = text_lc.find(term_lc, start)
                if idx == -1:
                    break
                end = idx + len(term_lc)

                # Skip if this span overlaps a previously accepted aspect
                overlap = any(not (end <= s or idx >= e) for (s, e) in used_spans)
                if overlap:
                    start = idx + 1
                    continue

                surface = original_text[idx:end]
                aspects.append(
                    {
                        "aspect": aspect_name,
                        "aspect_term": surface,
                        "category": category,
                        "pos": idx,
                    }
                )
                used_spans.append((idx, end))
                start = idx + 1

    return aspects

def _confidence_from_distance(dist: int) -> float:
    if dist <= 15:
        return 0.95
    if dist <= 40:
        return 0.9
    if dist <= 80:
        return 0.85
    return 0.8

# --------------------------
#  Public API
# --------------------------

def extract_aspect_opinions(
    text: str,
    sentiment_label: Optional[str] = None,
    emotion_label: Optional[str] = None,
) -> List[Dict]:
    """
    Main ABSA function used by actions.py.
    Returns list of dicts:
      {
        "aspect": canonical name,
        "aspect_term": surface form,
        "aspect_category": category,
        "opinion_term": surface opinion phrase or None,
        "aspect_sentiment": positive|negative|neutral,
        "aspect_sentiment_confidence": float,
      }
    """
    original_text = (text or "").strip()
    if not original_text:
        return []

    # Global fallback sentiment (from emotion pipeline)
    global_sentiment = None
    if sentiment_label in {"positive", "negative", "neutral"}:
        global_sentiment = sentiment_label
    elif emotion_label:
        coarse = coarse_from_label(emotion_label)
        if coarse in {"positive", "negative", "neutral"}:
            global_sentiment = coarse

    aspect_mentions = _find_all_aspect_mentions(original_text, original_text)
    opinion_mentions = _find_all_opinion_mentions(original_text, original_text)

    if not aspect_mentions:
        return []

    results: List[Dict] = []

    if opinion_mentions:
        pairs = []
        for ai, a in enumerate(aspect_mentions):
            for oi, o in enumerate(opinion_mentions):
                dist = abs(a["pos"] - o["pos"])
                pairs.append((dist, ai, oi))

        pairs.sort(key=lambda x: x[0])
        assigned_aspects = set()
        assigned_opinions = set()

        for dist, ai, oi in pairs:
            if ai in assigned_aspects or oi in assigned_opinions:
                continue
            assigned_aspects.add(ai)
            assigned_opinions.add(oi)

            a = aspect_mentions[ai]
            o = opinion_mentions[oi]
            conf = _confidence_from_distance(dist)

            results.append(
                {
                    "aspect": a["aspect"],
                    "aspect_term": a["aspect_term"],
                    "aspect_category": a["category"],
                    "opinion_term": o["opinion_term"],
                    "aspect_sentiment": o["sentiment"],
                    "aspect_sentiment_confidence": conf,
                }
            )

        # Aspects with no opinion get fallback sentiment
        for idx, a in enumerate(aspect_mentions):
            if idx in assigned_aspects:
                continue
            sent = global_sentiment or "neutral"
            conf = 0.7 if global_sentiment else 0.5
            results.append(
                {
                    "aspect": a["aspect"],
                    "aspect_term": a["aspect_term"],
                    "aspect_category": a["category"],
                    "opinion_term": None,
                    "aspect_sentiment": sent,
                    "aspect_sentiment_confidence": conf,
                }
            )
    else:
        # No opinion found at all, all aspects use global sentiment
        sent = global_sentiment or "neutral"
        conf = 0.7 if global_sentiment else 0.5
        for a in aspect_mentions:
            results.append(
                {
                    "aspect": a["aspect"],
                    "aspect_term": a["aspect_term"],
                    "aspect_category": a["category"],
                    "opinion_term": None,
                    "aspect_sentiment": sent,
                    "aspect_sentiment_confidence": conf,
                }
            )

    return results