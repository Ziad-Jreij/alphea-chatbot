"""
Criticality model benchmark: DeBERTa-v3-base-zeroshot-v2 vs BART-large-mnli.

Runs both models on the same 40-message test set with human ground-truth labels,
then reports per-class accuracy, macro-F1, latency, and head-to-head deltas.
The heuristic layer (keywords + structural rules) is intentionally bypassed so
we measure raw model quality, not guardrail quality.

Usage:
    python eval_criticality.py [--skip-bart]   # skip BART to save time/memory
"""

import sys
import time
import math
import statistics
import argparse
from typing import Optional

# ---------------------------------------------------------------------------
# Ground-truth test set  (40 messages, balanced across low/medium/high)
# ---------------------------------------------------------------------------
# Includes: clear-cut cases, edge cases, negation traps, hearsay, contrast
# structures, compound complaints, and polite inquiries.

TEST_CASES = [
    # ── LOW ──────────────────────────────────────────────────────────────────
    {"text": "Hi, what are your shipping times to London?",
     "label": "low"},
    {"text": "Can I change the color of my order before it ships?",
     "label": "low"},
    {"text": "Do you offer gift wrapping?",
     "label": "low"},
    {"text": "I just wanted to say thank you, everything arrived perfectly.",
     "label": "low"},
    {"text": "What is your return policy?",
     "label": "low"},
    {"text": "My order status says processing, is that normal?",
     "label": "low"},
    {"text": "Do you ship internationally?",
     "label": "low"},
    {"text": "How long does standard delivery take?",
     "label": "low"},
    {"text": "I'm wondering if this item is compatible with my phone.",
     "label": "low"},
    {"text": "I heard this might be a scam but I'm just checking.",  # hearsay trap
     "label": "low"},

    # ── MEDIUM ───────────────────────────────────────────────────────────────
    {"text": "My package is 3 days late, can you look into this?",
     "label": "medium"},
    {"text": "I received the wrong item in my order.",
     "label": "medium"},
    {"text": "The item I got is slightly different from what was shown online.",
     "label": "medium"},
    {"text": "I've been waiting a week for a refund and haven't heard anything.",
     "label": "medium"},
    {"text": "I ordered two items but only received one.",
     "label": "medium"},
    {"text": "The packaging was damaged when it arrived, but the product looks okay.",
     "label": "medium"},
    {"text": "I need to change my delivery address, the order hasn't shipped yet.",
     "label": "medium"},
    {"text": "The app keeps showing an error when I try to track my order.",
     "label": "medium"},
    {"text": "My discount code didn't apply at checkout even though it should be valid.",
     "label": "medium"},
    {"text": "I emailed support two days ago and haven't gotten a reply.",
     "label": "medium"},

    # ── HIGH ─────────────────────────────────────────────────────────────────
    {"text": "My card was charged twice and I need this fixed immediately!",
     "label": "high"},
    {"text": "The device started smoking and I turned it off — is this a safety issue?",
     "label": "high"},
    {"text": "I've reported this problem four times and no one is helping me.",
     "label": "high"},
    {"text": "Someone made a purchase on my account that I did not authorise.",
     "label": "high"},
    {"text": "I think my account has been hacked, please help urgently.",
     "label": "high"},
    {"text": "The product caught fire and I had to throw it out the window!",
     "label": "high"},
    {"text": "I'm extremely frustrated — this is the third time you've sent the wrong order.",
     "label": "high"},
    {"text": "I received medication with the wrong dosage label on it.",
     "label": "high"},
    {"text": "There's an unauthorized charge of $400 on my account from yesterday.",
     "label": "high"},
    {"text": "My child almost swallowed a small part that fell off your product.",
     "label": "high"},

    # ── EDGE / MIXED ─────────────────────────────────────────────────────────
    {"text": "The delivery was fine but the product stopped working after one day.",
     "label": "high"},   # contrast: fine BUT broken
    {"text": "Everything went smoothly, no complaints at all.",
     "label": "low"},
    {"text": "My card was NOT stolen, I'm just checking a charge.",  # negation trap
     "label": "low"},
    {"text": "I'm not saying there's fraud, just a charge I don't recognise.",  # negation
     "label": "medium"},
    {"text": "I've contacted you three times about a missing refund with no response.",
     "label": "high"},   # repeated frustration
    {"text": "The product is okay but the customer service was really disappointing.",
     "label": "medium"},
    {"text": "I want to return an item — it just isn't the right size.",
     "label": "low"},
    {"text": "Your app keeps crashing every time I open the checkout page.",
     "label": "high"},   # core-function failure
    {"text": "I accidentally ordered two, can you cancel one?",
     "label": "low"},
    {"text": "I've been a loyal customer for years and this experience has been terrible.",
     "label": "medium"},
]

# ---------------------------------------------------------------------------
# Models to benchmark
# ---------------------------------------------------------------------------
MODELS = {
    "DeBERTa-v3  (new)": "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    "BART-large  (old)": "facebook/bart-large-mnli",
}

CANDIDATE_LABELS = ["low urgency", "medium urgency", "high urgency"]
HYPOTHESIS      = "This support request is {}."
LABEL_MAP       = {"low urgency": "low", "medium urgency": "medium", "high urgency": "high"}

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_model(pipe, text: str) -> str:
    result = pipe(
        text,
        candidate_labels=CANDIDATE_LABELS,
        hypothesis_template=HYPOTHESIS,
        multi_label=False,
    )
    top = result["labels"][0]
    return LABEL_MAP[top]


def benchmark(model_name: str, label: str, cases: list) -> dict:
    from transformers import pipeline
    import torch

    print(f"\n{'─'*60}")
    print(f"  Loading: {label}  ({model_name})")
    print(f"{'─'*60}")

    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"
    print(f"  Device: {device_name}")

    t_load = time.time()
    pipe = pipeline("zero-shot-classification", model=model_name, device=device)
    load_time = time.time() - t_load
    print(f"  Model loaded in {load_time:.1f}s")

    predictions = []
    latencies   = []

    for case in cases:
        t0 = time.time()
        pred = run_model(pipe, case["text"])
        latencies.append(time.time() - t0)
        predictions.append({
            "text":  case["text"],
            "truth": case["label"],
            "pred":  pred,
            "ok":    pred == case["label"],
        })

    return {
        "label":       label,
        "model_name":  model_name,
        "predictions": predictions,
        "latencies":   latencies,
        "load_time":   load_time,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(predictions: list) -> dict:
    classes = ["low", "medium", "high"]
    tp = {c: 0 for c in classes}
    fp = {c: 0 for c in classes}
    fn = {c: 0 for c in classes}

    for p in predictions:
        t, pr = p["truth"], p["pred"]
        for c in classes:
            if t == c and pr == c:
                tp[c] += 1
            elif t != c and pr == c:
                fp[c] += 1
            elif t == c and pr != c:
                fn[c] += 1

    precision, recall, f1 = {}, {}, {}
    for c in classes:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec  = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1[c]        = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precision[c] = prec
        recall[c]    = rec

    correct = sum(1 for p in predictions if p["ok"])
    accuracy = correct / len(predictions)
    macro_f1 = sum(f1.values()) / len(classes)

    return {
        "accuracy":  accuracy,
        "macro_f1":  macro_f1,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "correct":   correct,
        "total":     len(predictions),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results: list):
    classes = ["low", "medium", "high"]
    sep = "═" * 72

    print(f"\n{sep}")
    print("  BENCHMARK RESULTS")
    print(sep)

    for r in results:
        m  = compute_metrics(r["predictions"])
        lats = r["latencies"]
        print(f"\n  {r['label']}")
        print(f"  {'─'*50}")
        print(f"  Accuracy  : {m['accuracy']*100:.1f}%  ({m['correct']}/{m['total']} correct)")
        print(f"  Macro-F1  : {m['macro_f1']*100:.1f}%")
        print(f"  Latency   : median {statistics.median(lats)*1000:.0f}ms  "
              f"p95 {sorted(lats)[int(len(lats)*0.95)]*1000:.0f}ms  "
              f"mean {statistics.mean(lats)*1000:.0f}ms")
        print(f"  Load time : {r['load_time']:.1f}s")
        print()
        print(f"  {'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'─'*42}")
        for c in classes:
            print(f"  {c:<8} {m['precision'][c]*100:>9.1f}% {m['recall'][c]*100:>9.1f}% {m['f1'][c]*100:>9.1f}%")

    # Head-to-head delta (only when both models ran)
    if len(results) == 2:
        m0 = compute_metrics(results[0]["predictions"])
        m1 = compute_metrics(results[1]["predictions"])
        d_acc = (m0["accuracy"] - m1["accuracy"]) * 100
        d_f1  = (m0["macro_f1"] - m1["macro_f1"]) * 100
        lat0  = statistics.median(results[0]["latencies"]) * 1000
        lat1  = statistics.median(results[1]["latencies"]) * 1000
        speedup = lat1 / lat0 if lat0 > 0 else 0

        print(f"\n{sep}")
        print("  HEAD-TO-HEAD: DeBERTa (new) vs BART (old)")
        print(sep)
        sign = lambda x: ("+" if x > 0 else "") + f"{x:.1f}"
        print(f"  Accuracy delta  : {sign(d_acc)} pp  "
              f"({'DeBERTa wins' if d_acc > 0 else 'BART wins' if d_acc < 0 else 'tie'})")
        print(f"  Macro-F1 delta  : {sign(d_f1)} pp")
        print(f"  Median latency  : DeBERTa {lat0:.0f}ms  BART {lat1:.0f}ms  "
              f"  →  {speedup:.1f}× {'faster' if speedup > 1 else 'slower'} per message")
        print()

    # Per-case breakdown
    print(f"\n{sep}")
    print("  PER-CASE BREAKDOWN  (ground truth → model predictions)")
    print(sep)
    header = f"  {'Truth':<7}"
    for r in results:
        col = r['label'].split()[0]   # "DeBERTa" or "BART"
        header += f"  {col:<8}"
    header += f"  Text"
    print(header)
    print(f"  {'─'*68}")

    for i, case in enumerate(TEST_CASES):
        truth = case["label"]
        row   = f"  {truth:<7}"
        any_wrong = False
        for r in results:
            pred = r["predictions"][i]["pred"]
            ok   = pred == truth
            mark = "  " if ok else "✗ "
            row += f"  {mark}{pred:<6}"
            if not ok:
                any_wrong = True
        trunc = case["text"][:52] + "…" if len(case["text"]) > 52 else case["text"]
        flag  = " ◀" if any_wrong else ""
        print(row + f"  {trunc}{flag}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bart", action="store_true",
                        help="Only run DeBERTa (faster, lower memory)")
    args = parser.parse_args()

    print("\n" + "═"*72)
    print("  CRITICALITY MODEL BENCHMARK")
    print("  40 labelled test messages  |  heuristics disabled (raw model only)")
    print("═"*72)

    to_run = list(MODELS.items())
    if args.skip_bart:
        to_run = [to_run[0]]  # only DeBERTa

    results = []
    for label, model_name in to_run:
        try:
            r = benchmark(model_name, label, TEST_CASES)
            results.append(r)
        except Exception as e:
            print(f"\n  ERROR loading {label}: {e}")

    if results:
        print_table(results)
    else:
        print("No models ran successfully.")


if __name__ == "__main__":
    main()
