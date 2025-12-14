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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
