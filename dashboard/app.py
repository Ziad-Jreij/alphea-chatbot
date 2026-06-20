import os
import csv
import io
import jwt
import time
from pathlib import Path
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import wraps
from collections import defaultdict, Counter

from flask import Flask, request, jsonify, Response, g
from flask_cors import CORS
from sqlalchemy import create_engine, Column, String, DateTime, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, sessionmaker

try:
    from prometheus_client import (
        Counter as PCounter, Histogram, Gauge,
        generate_latest, CONTENT_TYPE_LATEST,
    )
    _PROMETHEUS = True
except ImportError:
    _PROMETHEUS = False

from observability.logging_config import configure_logging, get_logger
configure_logging()
log = get_logger(__name__)

# ── Prometheus metrics (no-ops when prometheus_client not installed) ───────────
if _PROMETHEUS:
    _REQ_COUNT   = PCounter("vexa_requests_total",
                            "HTTP requests", ["endpoint", "method", "status"])
    _REQ_LATENCY = Histogram("vexa_request_duration_seconds",
                             "HTTP request duration", ["endpoint"])
    _SESS_TOTAL  = Gauge("vexa_sessions_total",      "Total chat sessions")
    _SESS_RED    = Gauge("vexa_sessions_red_alerts", "Red-alert sessions")
    _SESS_ESC    = Gauge("vexa_sessions_escalated",  "Escalated sessions")

# ── Config ────────────────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://chatbot:chatbot@localhost:5432/chatbot")
JWT_SECRET   = os.environ.get("JWT_SECRET_KEY", "dev-secret-change-in-prod")
ADMIN_USER   = os.environ.get("DASHBOARD_USERNAME", "admin")
ADMIN_PASS   = os.environ.get("DASHBOARD_PASSWORD", "admin")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
PORT         = int(os.environ.get("DASHBOARD_PORT", "5001"))

# ── Database ──────────────────────────────────────────────────────────────────
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

class Base(DeclarativeBase):
    pass

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    session_id = Column(String,  primary_key=True)
    data       = Column(JSONB,   nullable=False, default=dict)
    status     = Column(String(20), index=True,  default="green")
    updated_at = Column(DateTime(timezone=True), index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

try:
    Base.metadata.create_all(engine)
except Exception:
    # Another gunicorn worker won the race and already created the table; safe to ignore.
    engine.dispose()
_SessionFactory = sessionmaker(bind=engine)

@contextmanager
def get_db():
    db = _SessionFactory()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# ── Flask ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=CORS_ORIGINS, allow_headers=["Authorization", "Content-Type"])

@app.before_request
def _start_timer():
    g.t0 = time.monotonic()

@app.after_request
def _record_metrics(response):
    if _PROMETHEUS and hasattr(g, "t0"):
        endpoint = request.endpoint or "unknown"
        _REQ_COUNT.labels(endpoint=endpoint, method=request.method, status=response.status_code).inc()
        _REQ_LATENCY.labels(endpoint=endpoint).observe(time.monotonic() - g.t0)
    return response

# ── Rate limiting (in-memory, per IP) ────────────────────────────────────────
_rate_buckets: dict = defaultdict(list)
_last_bucket_sweep: float = 0.0
_SWEEP_INTERVAL = 300  # prune stale IPs every 5 minutes

def _is_rate_limited(key: str, max_calls: int, window_secs: int) -> bool:
    global _last_bucket_sweep
    now = time.monotonic()

    # Periodic global sweep to prevent unbounded dict growth
    if now - _last_bucket_sweep > _SWEEP_INTERVAL:
        stale = [k for k, ts in list(_rate_buckets.items()) if not ts or now - max(ts) > window_secs * 2]
        for k in stale:
            del _rate_buckets[k]
        _last_bucket_sweep = now

    calls = [t for t in _rate_buckets[key] if now - t < window_secs]
    _rate_buckets[key] = calls
    if len(calls) >= max_calls:
        return True
    _rate_buckets[key].append(now)
    return False

def rate_limit(max_calls: int = 60, window_secs: int = 60):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            ip = (
                request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
                .split(",")[0]
                .strip()
            )
            if _is_rate_limited(f"{f.__name__}:{ip}", max_calls, window_secs):
                return jsonify({"error": "too many requests"}), 429
            return f(*args, **kwargs)
        return decorated
    return decorator

# ── Auth ──────────────────────────────────────────────────────────────────────
def _issue_token() -> str:
    payload = {
        "sub": ADMIN_USER,
        "exp": datetime.now(timezone.utc) + timedelta(hours=12),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth  = request.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else request.args.get("token", "")
        if not token:
            return jsonify({"error": "missing token"}), 401
        try:
            jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        except jwt.InvalidTokenError as exc:
            return jsonify({"error": "invalid or expired token", "detail": str(exc)}), 401
        return f(*args, **kwargs)
    return decorated

@app.route("/health", methods=["GET"])
def health():
    """Unauthenticated health check — used by load balancers and docker-compose."""
    db_ok = False
    try:
        with get_db() as db:
            db.execute(text("SELECT 1"))
            db_ok = True
    except Exception as exc:
        log.error("health_check_db_failed", error=str(exc))

    status = "ok" if db_ok else "degraded"
    code   = 200 if db_ok else 503
    return jsonify({
        "status":    status,
        "database":  "ok" if db_ok else "unreachable",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }), code


@app.route("/metrics", methods=["GET"])
def metrics():
    """Prometheus scrape endpoint — unauthenticated (scraped by internal Prometheus)."""
    if not _PROMETHEUS:
        return jsonify({"error": "prometheus_client not installed"}), 501

    try:
        with get_db() as db:
            records = db.query(ChatSession).all()
            _SESS_TOTAL.set(len(records))
            _SESS_RED.set(sum(1 for r in records if r.status == "red"))
            _SESS_ESC.set(sum(
                1 for r in records
                if (r.data.get("conversation_escalation") or "").lower() == "escalate_now"
            ))
    except Exception as exc:
        log.error("metrics_db_failed", error=str(exc))

    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/api/auth/login", methods=["POST"])
@rate_limit(max_calls=5, window_secs=60)
def login():
    data = request.get_json(force=True) or {}
    if data.get("username") == ADMIN_USER and data.get("password") == ADMIN_PASS:
        log.info("login_success", user=ADMIN_USER)
        return jsonify({"access_token": _issue_token(), "token_type": "bearer"})
    log.warning("login_failed", user=data.get("username"))
    return jsonify({"error": "invalid credentials"}), 401

# ── Helpers ───────────────────────────────────────────────────────────────────
def _parse_iso_z(dt_str: str):
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None

def _compute_status(session: dict) -> str:
    if session.get("resolved") is True:
        return "green"
    criticality = (session.get("criticality") or "").lower()
    esc         = (session.get("escalation") or "none").lower()
    conv_esc    = (session.get("conversation_escalation") or "none").lower()
    handoff     = bool(session.get("handoff_requested"))
    if handoff or esc == "escalate_now" or conv_esc == "escalate_now" or criticality == "high":
        return "red"
    if criticality == "low":
        return "green"
    return "black"

def _rec_to_dict(rec: ChatSession) -> dict:
    d = dict(rec.data)
    d["session_id"] = rec.session_id
    return d

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/api/update_session", methods=["POST"])
@rate_limit(max_calls=300, window_secs=60)
def update_session():
    """Called by the Rasa action server (server-to-server, no JWT needed)."""
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    with get_db() as db:
        rec      = db.query(ChatSession).filter_by(session_id=session_id).first()
        existing = dict(rec.data) if rec else {}
        merged   = {**existing, **data}

        if not merged.get("updated_at"):
            merged["updated_at"] = datetime.now(timezone.utc).isoformat()

        merged["status"] = merged.get("status") or _compute_status(merged)
        updated_at = _parse_iso_z(merged.get("updated_at")) or datetime.now(timezone.utc)

        if rec:
            rec.data       = merged
            rec.status     = merged["status"]
            rec.updated_at = updated_at
        else:
            rec = ChatSession(
                session_id=session_id,
                data=merged,
                status=merged["status"],
                updated_at=updated_at,
            )
            db.add(rec)
        db.commit()

    log.debug("session_updated",
              session_id=session_id,
              status=merged["status"],
              escalation=merged.get("conversation_escalation"))
    return jsonify({"ok": True})


@app.route("/api/sessions", methods=["GET"])
@require_auth
@rate_limit(max_calls=120, window_secs=60)
def list_sessions():
    with get_db() as db:
        records = db.query(ChatSession).order_by(ChatSession.updated_at.desc().nullslast()).all()
        rows = []
        for rec in records:
            s = rec.data
            rows.append({
                "session_id":                     rec.session_id,
                "status":                         rec.status,
                "criticality":                    s.get("criticality"),
                "emotion":                        s.get("emotion"),
                "emotion_score":                  s.get("emotion_score"),
                "updated_at":                     s.get("updated_at"),
                "sentiment":                      s.get("sentiment"),
                "criticality_score":              s.get("criticality_score"),
                "escalation":                     s.get("escalation"),
                "conversation_criticality":       s.get("conversation_criticality"),
                "conversation_criticality_score": s.get("conversation_criticality_score"),
                "conversation_escalation":        s.get("conversation_escalation"),
                "handoff_requested":              s.get("handoff_requested"),
                "resolved":                       s.get("resolved", False),
                "last_user_message":              s.get("last_user_message"),
                "last_bot_message":               s.get("last_bot_message"),
            })
    return jsonify(rows)


@app.route("/api/sessions/<session_id>", methods=["GET"])
@require_auth
@rate_limit(max_calls=120, window_secs=60)
def get_session(session_id):
    with get_db() as db:
        rec = db.query(ChatSession).filter_by(session_id=session_id).first()
        if not rec:
            return jsonify({"error": "not found"}), 404
        d = _rec_to_dict(rec)
        d["status"] = rec.status or _compute_status(d)
        return jsonify(d)


@app.route("/api/sessions/<session_id>/resolve", methods=["POST"])
@require_auth
@rate_limit(max_calls=60, window_secs=60)
def resolve_session(session_id):
    with get_db() as db:
        rec = db.query(ChatSession).filter_by(session_id=session_id).first()
        if not rec:
            return jsonify({"error": "not found"}), 404

        merged = dict(rec.data)
        merged["resolved"]          = True
        merged["resolution_status"] = "resolved"
        merged["status"]            = "green"
        merged["updated_at"]        = datetime.now(timezone.utc).isoformat()

        rec.data       = merged
        rec.status     = "green"
        rec.updated_at = datetime.now(timezone.utc)
        db.commit()

        log.info("session_resolved", session_id=session_id)
        return jsonify({"ok": True, "session": merged})


@app.route("/api/stats", methods=["GET"])
@require_auth
@rate_limit(max_calls=60, window_secs=60)
def stats():
    with get_db() as db:
        records = db.query(ChatSession).all()
        total    = len(records)
        red      = sum(1 for r in records if r.status == "red")
        potential = sum(
            1 for r in records
            if (r.data.get("conversation_escalation") or "").lower() == "potential"
        )
    return jsonify({"total_sessions": total, "red_alerts": red, "potential_escalations": potential})


@app.route("/api/manager/overview", methods=["GET"])
@require_auth
@rate_limit(max_calls=60, window_secs=60)
def manager_overview():
    with get_db() as db:
        sessions = [r.data for r in db.query(ChatSession).all()]

    total     = len(sessions)
    escalated = resolved = abandoned = 0
    crit_scores      = []
    emotion_counts   = Counter()
    aspect_buckets   = defaultdict(lambda: {"mentions": 0, "negative": 0, "display": ""})
    timeline_buckets = Counter()

    for s in sessions:
        res_status = (s.get("resolution_status") or "").lower()
        conv_esc   = (s.get("conversation_escalation") or "none").lower()
        handoff    = bool(s.get("handoff_requested"))

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

        for key, d in (s.get("absa_insights") or {}).items():
            if key == "__summary__":
                continue
            biz     = d.get("business_aspect", key)
            display = d.get("display", key)
            aspect_buckets[biz]["mentions"] += d.get("mentions", 0)
            aspect_buckets[biz]["negative"] += d.get("negative", 0)
            aspect_buckets[biz]["display"]   = display

        updated_at = s.get("updated_at")
        if updated_at and (handoff or conv_esc == "escalate_now"):
            dt = _parse_iso_z(updated_at)
            if dt:
                bucket_min = (dt.minute // 30) * 30
                timeline_buckets[dt.strftime(f"%H:{bucket_min:02d}")] += 1

    top_neg = sorted(
        [
            {
                "business_aspect": biz,
                "display":   d["display"] or biz,
                "mentions":  d["mentions"],
                "negative":  d["negative"],
                "pct_neg":   round(d["negative"] / d["mentions"] * 100) if d["mentions"] else 0,
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

    return jsonify({
        "kpis": {
            "total_sessions":        total,
            "escalated":             escalated,
            "resolved":              resolved,
            "avg_criticality_score": round(sum(crit_scores) / len(crit_scores), 2) if crit_scores else 0.0,
            "resolution_rate_pct":   round(resolved / total * 100) if total else 0,
        },
        "lifecycle_counts": {
            "active":    max(0, total - escalated - resolved - abandoned),
            "escalated": escalated,
            "resolved":  resolved,
            "abandoned": abandoned,
        },
        "emotion_distribution": dict(emotion_counts.most_common()),
        "top_negative_aspects": top_neg,
        "escalation_timeline":  timeline,
    })


@app.route("/api/manager/export.csv", methods=["GET"])
@require_auth
@rate_limit(max_calls=10, window_secs=60)
def export_csv():
    fieldnames = [
        "session_id", "lifecycle", "status",
        "criticality", "criticality_score",
        "escalation", "conversation_escalation",
        "emotion", "sentiment",
        "top_negative_aspects",
        "updated_at",
    ]

    with get_db() as db:
        sessions = [(r.session_id, r.data, r.status) for r in db.query(ChatSession).all()]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for s_id, s, status in sessions:
        res_status = (s.get("resolution_status") or "").lower()
        conv_esc   = (s.get("conversation_escalation") or "none").lower()
        handoff    = bool(s.get("handoff_requested"))

        if res_status == "resolved":
            lifecycle = "Resolved"
        elif handoff or conv_esc == "escalate_now":
            lifecycle = "Escalated"
        elif res_status == "abandoned":
            lifecycle = "Abandoned"
        else:
            lifecycle = "Active"

        insights    = s.get("absa_insights") or {}
        neg_aspects = sorted(
            [
                d.get("display", k)
                for k, d in insights.items()
                if k != "__summary__" and d.get("dominant_sentiment") == "negative"
            ],
            key=lambda x: next(
                (v.get("negative", 0) for k, v in insights.items() if v.get("display") == x or k == x),
                0,
            ),
            reverse=True,
        )

        writer.writerow({
            "session_id":              s_id,
            "lifecycle":               lifecycle,
            "status":                  status or _compute_status(s),
            "criticality":             s.get("criticality", ""),
            "criticality_score":       s.get("conversation_criticality_score") or s.get("criticality_score", ""),
            "escalation":              s.get("escalation", ""),
            "conversation_escalation": s.get("conversation_escalation", ""),
            "emotion":                 s.get("emotion", ""),
            "sentiment":               s.get("sentiment", ""),
            "top_negative_aspects":    "; ".join(neg_aspects[:3]),
            "updated_at":              s.get("updated_at", ""),
        })

    return Response(
        output.getvalue().encode("utf-8"),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=sessions_export.csv"},
    )


# ── Static HTML file serving ──────────────────────────────────────────────────
_STATIC_DIR = Path(__file__).parent

@app.route("/")
def index():
    from flask import redirect
    return redirect("/login.html")

@app.route("/<path:filename>")
def serve_static(filename):
    from flask import send_from_directory, abort
    allowed = {f.name for f in _STATIC_DIR.glob("*.html")} | {f.name for f in _STATIC_DIR.glob("*.ico")}
    if filename not in allowed:
        abort(404)
    return send_from_directory(str(_STATIC_DIR), filename)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true",
    )
