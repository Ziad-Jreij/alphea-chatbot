"""
Criticality benchmark.

Scores criticality models on the labelled set in data/eval/criticality_benchmark.yml
in two modes:

  raw       the model alone, no guardrails or emotion prior — model quality
  pipeline  the production path (keyword guardrails → model → emotion prior →
            structural rules), i.e. infer_criticality_for_text() exactly as the
            action server calls it — measures what users actually get

and for either criticality strategy ("risk", the default, or legacy "urgency").

Splits: tune on --split dev; report --split holdout only for final numbers.

Usage (inside the rasa-actions image, models cached in the hf_cache volume):
    docker compose run --rm --no-deps -v "$PWD":/app -w /app rasa-actions \\
        python scripts/eval_criticality.py --json results/criticality.json

    python scripts/eval_criticality.py --models deberta-base --modes pipeline --show-errors
    python scripts/eval_criticality.py --strategies risk urgency  # compare strategies
    python scripts/eval_criticality.py --split holdout            # final reporting only
    python scripts/eval_criticality.py --source thesis            # original 40 cases only
"""

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import sys

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `components` imports when run as scripts/eval_criticality.py
BENCHMARK = ROOT / "data" / "eval" / "criticality_benchmark.yml"
CLASSES = ["low", "medium", "high"]

MODELS = {
    "deberta-base":   "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    "deberta-xsmall": "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
    "bart-large":     "facebook/bart-large-mnli",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_cases(source: str, split: str = "dev") -> list:
    cases = yaml.safe_load(BENCHMARK.read_text())["cases"]
    if split != "all":
        cases = [c for c in cases if c.get("split", "dev") == split]
    if source != "all":
        cases = [c for c in cases if c["source"] == source]
    return cases


def emotion_labels(cases: list) -> list:
    """Raw GoEmotions top label per case — the prior the action server uses."""
    from components.emotion import ensure_emotion_pipe

    pipe = ensure_emotion_pipe()
    out = []
    for c in cases:
        preds = pipe(c["text"])[0]
        out.append(max(preds, key=lambda p: p["score"])["label"])
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_model(model_key: str, cases: list, modes: list, emotions: list, strategy: str) -> dict:
    from components import criticality as crit

    model_name = MODELS[model_key]
    crit._crit_pipe = None               # ensure_crit_pipe caches the first model
    t_load = time.time()
    crit.ensure_crit_pipe(model_name)         # load once, timed separately from inference
    load_time = time.time() - t_load
    print(f"  {model_key}: loaded in {load_time:.1f}s")

    results = {}
    for mode in modes:
        preds, latencies = [], []
        for case, emo in zip(cases, emotions):
            t0 = time.time()
            if mode == "raw":
                res = crit.infer_criticality_for_text(case["text"], None, model_name=model_name,
                                                      strategy=strategy, boost_from_emotion=False,
                                                      heuristics_enabled=False)
            else:
                res = crit.infer_criticality_for_text(case["text"], emo, model_name=model_name,
                                                      strategy=strategy)
            pred, source = res["criticality_level"], res["source"]
            latencies.append(time.time() - t0)
            preds.append({**case, "pred": pred, "decided_by": source, "emotion": emo})
        results[mode] = {"predictions": preds, "latencies": latencies}
    return {"model": f"{model_key}/{strategy}", "model_name": model_name, "strategy": strategy,
            "load_time": load_time, "modes": results}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(predictions: list) -> dict:
    tp = {c: 0 for c in CLASSES}
    fp = {c: 0 for c in CLASSES}
    fn = {c: 0 for c in CLASSES}
    for p in predictions:
        t, pr = p["label"], p["pred"]
        for c in CLASSES:
            if t == c and pr == c:
                tp[c] += 1
            elif t != c and pr == c:
                fp[c] += 1
            elif t == c and pr != c:
                fn[c] += 1

    precision, recall, f1 = {}, {}, {}
    for c in CLASSES:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        precision[c], recall[c] = prec, rec
        f1[c] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    correct = sum(1 for p in predictions if p["label"] == p["pred"])
    return {
        "accuracy": correct / len(predictions) if predictions else 0.0,
        "macro_f1": sum(f1.values()) / len(CLASSES),
        "precision": precision, "recall": recall, "f1": f1,
        "correct": correct, "total": len(predictions),
    }


def grouped_accuracy(predictions: list, key) -> dict:
    groups = defaultdict(list)
    for p in predictions:
        for g in key(p):
            groups[g].append(p["label"] == p["pred"])
    return {g: (sum(v), len(v)) for g, v in sorted(groups.items())}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def report(runs: list, show_errors: bool) -> None:
    sep = "═" * 76
    print(f"\n{sep}\n  SUMMARY\n{sep}")
    print(f"  {'model':<22} {'mode':<9} {'acc':>6} {'macroF1':>8} {'HIGH-P':>7} {'HIGH-R':>7} "
          f"{'LOW→HIGH':>9} {'median ms':>10}")
    for run in runs:
        for mode, r in run["modes"].items():
            m = compute_metrics(r["predictions"])
            false_alarms = sum(1 for p in r["predictions"] if p["label"] == "low" and p["pred"] == "high")
            lat = statistics.median(r["latencies"]) * 1000
            print(f"  {run['model']:<22} {mode:<9} {pct(m['accuracy'])} {pct(m['macro_f1']):>8} "
                  f"{pct(m['precision']['high']):>7} {pct(m['recall']['high']):>7} "
                  f"{false_alarms:>9} {lat:>10.0f}")

    for run in runs:
        for mode, r in run["modes"].items():
            preds = r["predictions"]
            m = compute_metrics(preds)
            print(f"\n{sep}\n  {run['model']} — {mode}   ({m['correct']}/{m['total']} correct)\n{sep}")
            print(f"  {'class':<8} {'precision':>10} {'recall':>8} {'f1':>8}")
            for c in CLASSES:
                print(f"  {c:<8} {pct(m['precision'][c]):>10} {pct(m['recall'][c]):>8} {pct(m['f1'][c]):>8}")
            by_source = grouped_accuracy(preds, lambda p: [p["source"]])
            by_tag = grouped_accuracy(preds, lambda p: p.get("tags") or [])
            print("  by source: " + ", ".join(f"{k} {a}/{n}" for k, (a, n) in by_source.items()))
            if by_tag:
                print("  by trap:   " + ", ".join(f"{k} {a}/{n}" for k, (a, n) in by_tag.items()))
            if show_errors:
                print("  errors:")
                for p in preds:
                    if p["label"] != p["pred"]:
                        print(f"    {p['label']:>6} → {p['pred']:<6} [{p['decided_by']}] {p['text']}")


def write_json(runs: list, path: str) -> None:
    out = []
    for run in runs:
        for mode, r in run["modes"].items():
            m = compute_metrics(r["predictions"])
            out.append({
                "model": run["model"], "model_name": run["model_name"], "strategy": run["strategy"],
                "mode": mode, "n": len(r["predictions"]),
                "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                "median_latency_ms": statistics.median(r["latencies"]) * 1000,
                "low_to_high_false_alarms": sum(
                    1 for p in r["predictions"] if p["label"] == "low" and p["pred"] == "high"),
                "errors": [
                    {k: p.get(k) for k in ("text", "label", "pred", "decided_by", "source", "tags")}
                    for p in r["predictions"] if p["label"] != p["pred"]
                ],
            })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["deberta-base", "deberta-xsmall"], choices=MODELS)
    parser.add_argument("--modes", nargs="+", default=["raw", "pipeline"], choices=["raw", "pipeline"])
    parser.add_argument("--strategies", nargs="+", default=["risk"], choices=["risk", "urgency"])
    parser.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    parser.add_argument("--source", default="all", choices=["all", "thesis", "draft"])
    parser.add_argument("--no-emotion", action="store_true",
                        help="pipeline mode without the emotion prior (skips the emotion model)")
    parser.add_argument("--show-errors", action="store_true")
    parser.add_argument("--json", help="write machine-readable results to this path")
    args = parser.parse_args()

    cases = load_cases(args.source, args.split)
    print(f"\n  {len(cases)} cases (split={args.split}, source={args.source}) from {BENCHMARK.name}")

    emotions = [None] * len(cases)
    if "pipeline" in args.modes and not args.no_emotion:
        print("  scoring emotion prior …")
        emotions = emotion_labels(cases)

    runs = [run_model(m, cases, args.modes, emotions, st) for m in args.models for st in args.strategies]
    report(runs, args.show_errors)
    if args.json:
        write_json(runs, args.json)


if __name__ == "__main__":
    main()
