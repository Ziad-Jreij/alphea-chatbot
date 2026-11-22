import os
import csv
import hmac
import io
import jwt
import time
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import wraps
from collections import defaultdict, Counter

from flask import Flask, request, jsonify, Response, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from sqlalchemy import create_engine, Column, String, DateTime, JSON, text
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
    _REQ_COUNT   = PCounter("alphea_requests_total",
                            "HTTP requests", ["endpoint", "method", "status"])
    _REQ_LATENCY = Histogram("alphea_request_duration_seconds",
                             "HTTP request duration", ["endpoint"])
    _SESS_TOTAL  = Gauge("alphea_sessions_total",      "Total chat sessions")
    _SESS_RED    = Gauge("alphea_sessions_red_alerts", "Red-alert sessions")
    _SESS_ESC    = Gauge("alphea_sessions_escalated",  "Escalated sessions")

# ── Config ────────────────────────────────────────────────────────────────────
DATABASE_URL  = os.environ.get("DATABASE_URL", "postgresql://chatbot:chatbot@localhost:5432/chatbot")
JWT_SECRET    = os.environ.get("JWT_SECRET_KEY", "")
ADMIN_USER    = os.environ.get("DASHBOARD_USERNAME", "admin")
ADMIN_PASS    = os.environ.get("DASHBOARD_PASSWORD", "")
INGEST_TOKEN  = os.environ.get("DASHBOARD_INGEST_TOKEN", "")
CORS_ORIGINS  = os.environ.get("CORS_ORIGINS", "*").split(",")
PORT          = int(os.environ.get("DASHBOARD_PORT", "5001"))
# Number of reverse proxies in front of the app whose X-Forwarded-For we trust.
# 0 (default) = use the socket peer address; never trust client-supplied headers.
TRUSTED_PROXY_COUNT = int(os.environ.get("TRUSTED_PROXY_COUNT", "0"))
RATELIMIT_STORAGE   = os.environ.get("RATELIMIT_STORAGE_URI", "memory://")

# ── Refuse to start with missing or placeholder secrets ───────────────────────
_PLACEHOLDER_SECRETS = {
    "", "admin", "changeme", "password", "chatbot",
    "dev-secret-change-in-prod",
    "change-me-to-a-long-random-string-in-production",
}

def _check_secrets() -> None:
    problems = []
    if JWT_SECRET in _PLACEHOLDER_SECRETS or len(JWT_SECRET) < 32:
        problems.append("JWT_SECRET_KEY must be a random string of at least 32 characters")
    if ADMIN_PASS in _PLACEHOLDER_SECRETS or len(ADMIN_PASS) < 12:
        problems.append("DASHBOARD_PASSWORD must be set to a non-default value of at least 12 characters")
    if INGEST_TOKEN in _PLACEHOLDER_SECRETS or len(INGEST_TOKEN) < 32:
        problems.append("DASHBOARD_INGEST_TOKEN must be a random string of at least 32 characters")
    if problems:
        raise RuntimeError(
            "Insecure dashboard configuration:\n  - " + "\n  - ".join(problems)
            + "\nGenerate values with: openssl rand -hex 32"
        )

_check_secrets()

# ── Database ──────────────────────────────────────────────────────────────────
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

class Base(DeclarativeBase):
    pass

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    session_id = Column(String,  primary_key=True)
    # JSONB on PostgreSQL; plain JSON elsewhere (SQLite in unit tests).
    data       = Column(JSON().with_variant(JSONB(), "postgresql"), nullable=False, default=dict)
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
if TRUSTED_PROXY_COUNT > 0:
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=TRUSTED_PROXY_COUNT)
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

# ── Rate limiting (per client IP) ─────────────────────────────────────────────
# Use a shared store (RATELIMIT_STORAGE_URI=redis://...) so limits hold across
# gunicorn workers; memory:// is per-process and only suitable for development.
# The client IP comes from request.remote_addr, which ProxyFix rewrites only
# when TRUSTED_PROXY_COUNT > 0.
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri=RATELIMIT_STORAGE,
    strategy="moving-window",
)

@app.errorhandler(429)
def _too_many_requests(_exc):
    return jsonify({"error": "too many requests"}), 429

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
        token = auth[7:] if auth.startswith("Bearer ") else ""
        if not token:
            return jsonify({"error": "missing token"}), 401
        try:
            jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        except jwt.InvalidTokenError:
            return jsonify({"error": "invalid or expired token"}), 401
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


def _safe_equals(supplied, expected: str) -> bool:
    if not isinstance(supplied, str):
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))

@app.route("/api/auth/login", methods=["POST"])
@limiter.limit("5 per minute")
def login():
    data = request.get_json(force=True) or {}
    # Evaluate both comparisons so timing doesn't reveal which field was wrong.
    user_ok = _safe_equals(data.get("username"), ADMIN_USER)
    pass_ok = _safe_equals(data.get("password"), ADMIN_PASS)
    if user_ok and pass_ok:
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
@limiter.exempt
def update_session():
    """Called by the Rasa action server (server-to-server).

    Authenticated with the shared DASHBOARD_INGEST_TOKEN instead of a JWT. Not
    rate-limited: every chat turn from every user arrives from the single
    action-server IP, so a per-IP limit would throttle the bot under load.
    """
    if not _safe_equals(request.headers.get("X-Ingest-Token"), INGEST_TOKEN):
        log.warning("ingest_rejected", remote_addr=request.remote_addr)
        return jsonify({"error": "invalid ingest token"}), 401

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
@limiter.limit("120 per minute")
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
@limiter.limit("120 per minute")
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
@limiter.limit("60 per minute")
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
@limiter.limit("60 per minute")
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
@limiter.limit("60 per minute")
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
@limiter.limit("10 per minute")
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
