from flask import Flask, request, jsonify
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow dashboard.html to call the API from anywhere (demo-friendly)

# In-memory store for demo: {session_id: session_data}
SESSIONS = {}


def _parse_iso_z(dt_str: str):
    """Parse '2025-12-14T10:00:00.000Z' safely; return None if invalid."""
    if not dt_str:
        return None
    try:
        s = dt_str.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _compute_status(session: dict) -> str:
    """
    If status not provided, compute a best-effort status using your logic.
    This makes the backend robust even if the client forgets to send "status".
    """
    # If agent marked resolved, keep it non-red
    if session.get("resolved") is True:
        return "green"

    criticality = (session.get("criticality") or "").lower()
    esc = (session.get("escalation") or "none").lower()
    conv_esc = (session.get("conversation_escalation") or "none").lower()
    handoff = bool(session.get("handoff_requested"))

    if handoff or esc == "escalate_now" or conv_esc == "escalate_now" or criticality == "high":
        return "red"
    if criticality == "low":
        return "green"
    return "black"


@app.route("/api/update_session", methods=["POST"])
def update_session():
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    # Merge / overwrite existing data
    existing = SESSIONS.get(session_id, {})
    merged = {**existing, **data}

    # Normalize updated_at if missing
    if not merged.get("updated_at"):
        merged["updated_at"] = datetime.utcnow().isoformat() + "Z"

    # Ensure status exists (either use incoming or compute)
    merged["status"] = merged.get("status") or _compute_status(merged)

    SESSIONS[session_id] = merged
    return jsonify({"ok": True})


@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    """
    Returns sessions with the fields your dashboard uses.
    Sorts by updated_at desc (newest first).
    """
    rows = []
    for s_id, s_data in SESSIONS.items():
        rows.append({
            "session_id": s_id,

            # Side list fields
            "status": s_data.get("status"),
            "criticality": s_data.get("criticality"),
            "emotion": s_data.get("emotion"),
            "updated_at": s_data.get("updated_at"),

            # Extra fields used in your sidebar badges / details
            "sentiment": s_data.get("sentiment"),
            "criticality_score": s_data.get("criticality_score"),
            "escalation": s_data.get("escalation"),
            "conversation_criticality": s_data.get("conversation_criticality"),
            "conversation_criticality_score": s_data.get("conversation_criticality_score"),
            "conversation_escalation": s_data.get("conversation_escalation"),
            "handoff_requested": s_data.get("handoff_requested"),
            "resolved": s_data.get("resolved", False),

            # Preview lines
            "last_user_message": s_data.get("last_user_message"),
            "last_bot_message": s_data.get("last_bot_message"),
        })

    # Sort by time (newest first); invalid timestamps go last
    rows.sort(
        key=lambda r: _parse_iso_z(r.get("updated_at")) or datetime.min,
        reverse=True
    )
    return jsonify(rows)


@app.route("/api/sessions/<session_id>", methods=["GET"])
def get_session(session_id):
    s = SESSIONS.get(session_id)
    if not s:
        return jsonify({"error": "not found"}), 404
    # Guarantee status in response
    s = {**s, "status": s.get("status") or _compute_status(s)}
    return jsonify(s)


@app.route("/api/sessions/<session_id>/resolve", methods=["POST"])
def resolve_session(session_id):
    """
    Agent marks a conversation as resolved.
    We keep it in-memory (no expiry), but set resolved flag + status.
    """
    s = SESSIONS.get(session_id)
    if not s:
        return jsonify({"error": "not found"}), 404

    s["resolved"] = True
    s["resolution_status"] = "resolved"
    s["status"] = "green"
    s["updated_at"] = datetime.utcnow().isoformat() + "Z"
    SESSIONS[session_id] = s
    return jsonify({"ok": True, "session": s})


@app.route("/api/stats", methods=["GET"])
def stats():
    """
    Lightweight counts to display in the UI if you want.
    """
    total = len(SESSIONS)
    red = 0
    potential = 0
    for s in SESSIONS.values():
        st = s.get("status") or _compute_status(s)
        if st == "red":
            red += 1
        if (s.get("conversation_escalation") or "").lower() == "potential":
            potential += 1
    return jsonify({
        "total_sessions": total,
        "red_alerts": red,
        "potential_escalations": potential,
    })



@app.route("/api/manager/overview", methods=["GET"])
def manager_overview():
    """
    Cross-session aggregate for the Manager Overview screen.

    Returns:
      kpis                  - total, escalated, resolved, avg_criticality_score
      lifecycle_counts      - {active, escalated, resolved, abandoned}
      emotion_distribution  - {emotion: count}
      top_negative_aspects  - [{display, mentions, negative, pct_neg}] sorted desc
      escalation_timeline   - [{bucket: "HH:MM", escalations: N}] last 12 half-hours
    """
    from collections import defaultdict, Counter

    total = len(SESSIONS)
    escalated = 0
    resolved = 0
    abandoned = 0
    crit_scores = []

    emotion_counts = Counter()
    aspect_buckets = defaultdict(lambda: {"mentions": 0, "negative": 0, "display": ""})
    timeline_buckets = Counter()

    for s in SESSIONS.values():
        res_status = (s.get("resolution_status") or "").lower()
        conv_esc = (s.get("conversation_escalation") or "none").lower()
        handoff = bool(s.get("handoff_requested"))

        if res_status == "resolved":
            resolved += 1
        elif handoff or conv_esc == "escalate_now":
            escalated += 1
        elif res_status == "abandoned":
            abandoned += 1

        cs = s.get("conversation_criticality_score") or s.get("criticality_score")
        if cs is not None:
            try:
                crit_scores.append(float(cs))
            except (ValueError, TypeError):
                pass

        emo = (s.get("emotion") or "").lower()
        if emo:
            emotion_counts[emo] += 1

        insights = s.get("absa_insights") or {}
        for key, d in insights.items():
            if key == "__summary__":
                continue
            biz = d.get("business_aspect", key)
            display = d.get("display", key)
            aspect_buckets[biz]["mentions"] += d.get("mentions", 0)
            aspect_buckets[biz]["negative"] += d.get("negative", 0)
            aspect_buckets[biz]["display"] = display

        updated_at = s.get("updated_at")
        if updated_at and (handoff or conv_esc == "escalate_now"):
            dt = _parse_iso_z(updated_at)
            if dt:
                bucket_min = (dt.minute // 30) * 30
                bucket_key = dt.strftime(f"%H:{bucket_min:02d}")
                timeline_buckets[bucket_key] += 1

    top_neg = sorted(
        [
            {
                "business_aspect": biz,
                "display": d["display"] or biz,
                "mentions": d["mentions"],
                "negative": d["negative"],
                "pct_neg": round(d["negative"] / d["mentions"] * 100) if d["mentions"] else 0,
            }
            for biz, d in aspect_buckets.items()
            if d["negative"] > 0
        ],
        key=lambda x: x["negative"],
        reverse=True,
    )[:7]

    timeline = [
        {"bucket": k, "escalations": v}
        for k, v in sorted(timeline_buckets.items())
    ][-12:]

    active = total - escalated - resolved - abandoned

    return jsonify({
        "kpis": {
            "total_sessions":        total,
            "escalated":             escalated,
            "resolved":              resolved,
            "avg_criticality_score": round(sum(crit_scores) / len(crit_scores), 2) if crit_scores else 0.0,
            "resolution_rate_pct":   round(resolved / total * 100) if total else 0,
        },
        "lifecycle_counts": {
            "active":    max(0, active),
            "escalated": escalated,
            "resolved":  resolved,
            "abandoned": abandoned,
        },
        "emotion_distribution":  dict(emotion_counts.most_common()),
        "top_negative_aspects":  top_neg,
        "escalation_timeline":   timeline,
    })


@app.route("/api/manager/export.csv", methods=["GET"])
def export_csv():
    """Full CSV export of all sessions for manager download."""
    import csv
    import io
    from flask import Response

    fieldnames = [
        "session_id", "lifecycle", "status",
        "criticality", "criticality_score",
        "escalation", "conversation_escalation",
        "emotion", "sentiment",
        "top_negative_aspects",
        "updated_at",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for s_id, s in SESSIONS.items():
        res_status = (s.get("resolution_status") or "").lower()
        conv_esc = (s.get("conversation_escalation") or "none").lower()
        handoff = bool(s.get("handoff_requested"))

        if res_status == "resolved":
            lifecycle = "Resolved"
        elif handoff or conv_esc == "escalate_now":
            lifecycle = "Escalated"
        elif res_status == "abandoned":
            lifecycle = "Abandoned"
        else:
            lifecycle = "Active"

        insights = s.get("absa_insights") or {}
        neg_aspects = sorted(
            [
                d.get("display", k)
                for k, d in insights.items()
                if k != "__summary__" and d.get("dominant_sentiment") == "negative"
            ],
            key=lambda x: next(
                (v.get("negative", 0) for k, v in insights.items()
                 if v.get("display") == x or k == x),
                0,
            ),
            reverse=True,
        )

        writer.writerow({
            "session_id":              s_id,
            "lifecycle":               lifecycle,
            "status":                  s.get("status") or _compute_status(s),
            "criticality":             s.get("criticality", ""),
            "criticality_score":       s.get("conversation_criticality_score") or s.get("criticality_score", ""),
            "escalation":              s.get("escalation", ""),
            "conversation_escalation": s.get("conversation_escalation", ""),
            "emotion":                 s.get("emotion", ""),
            "sentiment":               s.get("sentiment", ""),
            "top_negative_aspects":    "; ".join(neg_aspects[:3]),
            "updated_at":              s.get("updated_at", ""),
        })

    csv_bytes = output.getvalue().encode("utf-8")
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=sessions_export.csv"},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)