# components/absa.py

from __future__ import annotations
from typing import List, Dict, Optional
from components.emotion import coarse_from_label  # reuse if we need global sentiment fallback

# ---- Aspect taxonomy for generic product feedback ----
# name: what you store in "Aspect"
# terms: how it might appear in text
# category: high-level category for thesis (Performance, Features, Cost, Aesthetics, etc.)
ASPECT_SPECS = [
    {
    "name": "account security",
    "terms": [
        "account security",
        "suspicious transactions",
        "transactions i didn't make",
        "transactions I didn't make",
        "transactions i did not make",
        "charges i don't recognize",
        "charges i do not recognize",
        "unauthorized transactions",
        "unauthorized charges",
        "fraudulent transaction",
        "fraudulent charge",
        "card was used without my permission",
    ],
    "category": "Security",
    },
    {
    "name": "safety & defects",
    "terms": [
        "safety",
        "safe to use",
        "unsafe",
        "dangerous",
        "fire hazard",
        "caught fire",
        "on fire",
        "burned",
        "smelled like smoke",
        "smell of smoke",
        "leaking water",
        "leaking everywhere",
        "leaks everywhere",
        "water everywhere",
        "leaking",
        "leak",
        "electric shock",
        "gave me a shock",
        "short circuit",
    ],
    "category": "Reliability",
    },

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
            "stability",
            "stable",
            "unstable",
            "uptime",
        ],
        "category": "Reliability",
    },
    {
        "name": "delivery tracking",
        "terms": [
            "delivery tracker",
            "delivery tracking",
            "tracking",
            "track my delivery",
            "tracking page",
            "tracking link",
        ],
        "category": "Digital_Experience",
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
            "user interface",
            "interface",
            "ui",
            "ui design",
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
            "return shipping fee",
            "shipping fee",
            "return fee",
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
            "app",
            "the app",
            "application",
            "mobile app",
            "phone app",
            "the app experience",
            "in the app",
            "app interface",
            "app design",
            "app layout",
            "in the app",
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
    {
        "name": "refund speed",
        "terms": [
            "refund speed",
            "speed of the refund",
            "refund was fast",
            "refund was quick",
            "fast refund",
            "quick refund",
        ],
        "category": "Policy",
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
    "so happy",
    "really happy",
    "i'm very happy", "still great", "surprisingly fast", "very fast", 
    "looks polished", "clean", "clean and intuitive"
]


NEG_OPINIONS = [
    # Strong negative
    "bad", "very bad", "terrible", "awful", "horrible", "poor",
    "horrendous", "unacceptable", "feels cheap",
    "poor value",
    "not worth it", "outrageous",
    "return was difficult",
    "overall i'm unhappy", "unjustifiable", "not justifiable",

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
    "battery dies fast",
    "battery dies after",
    "dies after two hours",
    "dies after a few hours",
    "keeps freezing",
    "keeps crashing",
    "crashes often",
    "freezes randomly",
    "leaking water everywhere",
    "leaking water",
    "leaks everywhere",

    # Safety / medical
    "caught fire",
    "caught on fire",
    "wrong label",
    "wrong medicine",
    "wrong dosage",
    "dangerous",
    "not safe",
    "unsafe",
    "buggy",
    "crashes",
    "crashing",
    "crashes often",
    "keeps crashing",
    "freezes",
    "keeps freezing",

    # Security / fraud
    "transactions i didn't make",
    "transactions I didn't make",
    "unauthorized transactions",
    "unauthorized charges",
    "fraudulent charge",
    "fraudulent transaction", 
    # Mixed-clause negatives
    "crashes often",
    "keeps crashing",
    "keeps freezing",
    "freezes often",
    "freezes randomly",

    # Price
    "a little too high",
    "too high",
    "way too high",

    # Billing / correctness
    "incorrect for three months",
    "incorrect for months",
    "incorrect for three months now",
    "has been incorrect",
    "billing has been incorrect",

    # Performance / speed
    "loads incredibly slow",
    "loads really slow",
    "loads very slow",
    "incredibly slow",
    "really slow", "rude"
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
        for phrase in sorted(phrases, key=len, reverse=True):
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

        for term in sorted(spec["terms"], key=len, reverse=True):
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
    # Skip Rasa payload commands like "/cancel_order" or "/report_damaged_product"
    if original_text.startswith("/"):
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
    # --- Contrast / second-clause detection (for mixed messages) ---
    CONTRAST_MARKERS = [" but ", " even though ", " although ", " though ", " however "]
    NEG_CONTEXT_HINTS = [
        "too high", "unjustifiable", "outrageous", "ridiculous",
        "incorrect", "wrong", "broken", "not working", "doesn't work", "doesnt work",
        "slow", "lags", "laggy", "freezes", "crashes", "buggy",
        "unhelpful", "rude", "late", "delayed",
    ]

    text_lc = original_text.lower()
    second_clause_negative = False
    second_clause_start = None

    for marker in CONTRAST_MARKERS:
        idx = text_lc.find(marker)
        if idx != -1:
            second_clause_start = idx + len(marker)
            after = text_lc[second_clause_start:]
            if any(h in after for h in NEG_CONTEXT_HINTS):
                second_clause_negative = True
            break
    if not aspect_mentions:
        # If there is global sentiment, create a synthetic "overall experience" aspect
        if global_sentiment is not None:
            return [
                {
                    "aspect": "overall experience",
                    "aspect_term": None,
                    "aspect_category": "Overall_Experience",
                    "opinion_term": opinion_mentions[0]["opinion_term"]
                    if opinion_mentions
                    else None,
                    "aspect_sentiment": global_sentiment,
                    "aspect_sentiment_confidence": 0.7,
                }
            ]
        # No aspect and no sentiment -> nothing to log
        return []
    results: List[Dict] = []

    if opinion_mentions:
        pairs = []
        for ai, a in enumerate(aspect_mentions):
            for oi, o in enumerate(opinion_mentions):
                # penalize pairing across sentence boundaries
                same_sentence_bonus = 0
                a_before = original_text.rfind(".", 0, a["pos"])
                a_after  = original_text.find(".", a["pos"])
                o_before = original_text.rfind(".", 0, o["pos"])
                o_after  = original_text.find(".", o["pos"])

                if a_before == o_before and a_after == o_after:
                    same_sentence_bonus = -30  # pull same-sentence pairs closer

                dist = abs(a["pos"] - o["pos"]) + same_sentence_bonus
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

            # If this aspect is in the second clause and the second clause is negative,
            # override the fallback sentiment to negative.
            if second_clause_negative and second_clause_start is not None:
                if a["pos"] >= second_clause_start:
                    sent = "negative"
                    conf = 0.85

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