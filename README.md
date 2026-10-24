# Alphea — AI Customer Service Chatbot

Alphea is a production-ready e-commerce customer service chatbot built on Rasa 3.x. It combines three custom NLU components — real-time emotion detection, zero-shot criticality scoring, and aspect-based sentiment analysis — with a Flask admin dashboard that lets support teams monitor live conversations, track escalations, and export session data. The full stack runs in Docker with PostgreSQL persistence and a Prometheus/Grafana monitoring layer.

---

## Why This Project Is Novel

Most support bots stop at intent recognition or a single sentiment score. Alphea combines **emotion awareness, criticality assessment, escalation logic and aspect-level analysis** in real time to decide when a human agent should step in:

- Fine-grained **emotion detection** (GoEmotions-based)
- **Criticality scoring independent of emotion** — "I think someone stole my credit card" is emotionally neutral but high-criticality
- Hybrid **rule-based and model-based escalation**, with a reason code for every decision
- **Aspect-level sentiment** for products and services
- A **real-time dashboard** for human agents
- **Two automatically generated datasets** for analytics and business insight

The result is a bot that doesn't just chat: it monitors conversation risk, protects users in high-criticality situations, and turns support conversations into product feedback.

---

## Architecture

```
                        ┌────────────────────────────────────────────────────┐
                        │                  Docker Network                     │
                        │                                                      │
  Browser / Widget ────▶│  Rasa Dialog Server  :5005                          │
                        │  • NLU: BERT features → DIETClassifier (intents)     │
                        │  • dialogue policies, forms                          │
                        │  • HTTP API protected by RASA_AUTH_TOKEN             │
                        │                     │ webhook                        │
                        │                     ▼                                │
                        │  Rasa Actions Server  :5055 (internal)               │
                        │  • emotion detection (DistilBERT)                    │
                        │  • criticality scoring (DeBERTa zero-shot)           │
                        │  • ABSA per turn (MiniLM)                            │
                        │  • order lookups (mock / REST)                       │
                        │  • pushes session state → Dashboard (ingest token)   │
                        │                     │                                │
                        │          ┌──────────┴──────────┐                     │
                        │          │                     │                     │
                        │          ▼                     ▼                     │
                        │  PostgreSQL  :5432     Dashboard (Flask)  :5001      │
                        │  • Rasa tracker store  • JWT-protected API           │
                        │  • chat_sessions table • session monitoring          │
                        │                        • manager overview + CSV      │
                        │                                                      │
                        │  Prometheus  :9090     Grafana  :3000                │
                        │  scrapes /metrics      visualises all metrics        │
                        └────────────────────────────────────────────────────┘
```

---

## Prerequisites

- **Docker** ≥ 24 and **Docker Compose** ≥ 2.20
- **4 GB RAM** minimum (models load into the actions container; 6 GB recommended)
- A trained Rasa model `.tar.gz` in `./models/` — see [Training](#training) (the directory is gitignored, so a fresh clone has none)
- Internet access on first startup so the actions container can download model weights from HuggingFace Hub (~700 MB total for all three AI models; cached in the `hf_cache` Docker volume)

---

## Quick Start

```bash
git clone https://github.com/Ziad-Jreij/alphea-chatbot.git
cd alphea-chatbot

# Copy the example env file and fill in the four required secrets.
# The dashboard refuses to start while they are empty or set to defaults.
cp .env.example .env
for k in JWT_SECRET_KEY DASHBOARD_INGEST_TOKEN RASA_AUTH_TOKEN; do
  sed -i "s/^$k=.*/$k=$(openssl rand -hex 32)/" .env
done
# ...and set DASHBOARD_PASSWORD (12+ characters) yourself.

# Train a model on first run (see Training below)
mkdir -p models
docker-compose run --rm --no-deps --build --user "$(id -u):0" rasa train

# Bring up the full stack (postgres → dashboard → rasa-actions → rasa → monitoring)
docker-compose up --build
```

Services come up in dependency order. On first run the actions container downloads ~700 MB of model weights into the `hf_cache` volume before it reports healthy, and Rasa waits for it — expect several minutes on a slow connection. Later starts take seconds.

| Service | URL | Exposure |
|---|---|---|
| Rasa (chat + HTTP API) | http://localhost:5005 | public — API needs `?token=$RASA_AUTH_TOKEN` |
| Admin Dashboard | http://localhost:5001 | public — JWT login |
| Action Server | http://localhost:5055 | bound to 127.0.0.1 only |
| PostgreSQL | localhost:5432 | bound to 127.0.0.1 only |
| Prometheus | http://localhost:9090 | bound to 127.0.0.1 only |
| Grafana | http://localhost:3000 | bound to 127.0.0.1 only |

Log in to the dashboard at http://localhost:5001 with the credentials set in `DASHBOARD_USERNAME` / `DASHBOARD_PASSWORD`.

### Training

Train inside the Rasa container whenever `domain.yml`, `config.yml` or anything under `data/` changes — Rasa serves the domain baked into the model, so edits are invisible until you retrain:

```bash
docker-compose run --rm --no-deps --build --user "$(id -u):0" rasa train
```

`--build` matters: training data is copied into the image. The model is written to `./models/` and the newest one is loaded on the next `docker-compose up`. `--user "$(id -u):0"` runs training as your host user so it can write the model into `./models`. The first run downloads BERT weights (~440 MB) into the `rasa_hf_cache` volume; later runs reuse them.

---

## Performance & Hardware

### Default behaviour

The AI models run on **GPU if one is available, CPU otherwise** — no configuration required. The `GPU_ENABLED=auto` default in `.env.example` handles this automatically. If you deploy on a machine without a GPU, the stack will start and work correctly; messages will just take longer to score (see the latency table below).

### Model benchmark — criticality component

The criticality component has been benchmarked on 40 labelled customer-service messages with the heuristic guardrail layer **disabled** (raw model output only). Real-world accuracy is higher because the keyword and safety-rule layers intercept the most obvious high-urgency messages before the model is called.

| Model | Accuracy | Macro-F1 | CPU latency | Size | Status |
|---|---|---|---|---|---|
| **DeBERTa-v3-base-zeroshot-v2.0** | **60.0%** | **58.6%** | ~1500 ms | 184 MB | ✅ Current |
| BART-large-mnli | 37.5% | 31.6% | ~400 ms | 1.6 GB | ❌ Replaced |
| DeBERTa-v3-xsmall | ~65–70% | — | ~450 ms | 70 MB | CPU alternative |
| xtremedistil-l6 | ~50% | — | ~27 ms | 25 MB | Too inaccurate |

**Why DeBERTa-v3-base was chosen over BART:** BART had a severe bias toward predicting "medium" for almost every message (84.6% medium recall, only 15.4% high recall), making it nearly useless for catching genuine escalations. DeBERTa's high-class precision is 88.9% — when it says escalate, it is almost always right.

**Per-class breakdown for the current model (DeBERTa-v3-base, heuristics off):**

| Class | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| LOW | 52.2% | 85.7% | 64.9% | Good at clearing routine queries |
| MEDIUM | 50.0% | 30.8% | 38.1% | Weakest class; keyword layer compensates |
| **HIGH** | **88.9%** | **61.5%** | **72.7%** | Most critical — high precision, few false alarms |

With the heuristic layer enabled (production default), HIGH recall rises above 90% because safety, finance, and medical keywords are caught before the model runs.

### CPU vs GPU latency

| Hardware | Median latency / message | Recommended for |
|---|---|---|
| CPU only | ~1500 ms | Development, low-traffic demos (< 5 concurrent users) |
| NVIDIA GPU (e.g. T4) | ~30–50 ms | Production, > 10 concurrent users |

**CPU is usable** for development and low-traffic demos: each user turn is scored once by the action server, so a ~1.5 s criticality pass is added to the bot's reply time. Switch to the xsmall model (below) or a GPU before putting real traffic on it.

**GPU is recommended for production.** At ~40 ms per message the latency is invisible to users regardless of traffic.

### Controlling GPU usage

Set `GPU_ENABLED` in your `.env`:

| Value | Behaviour |
|---|---|
| `auto` *(default)* | Uses GPU when CUDA is available; silently falls back to CPU if not. |
| `true` | Requires a GPU — the actions container **refuses to start** without one. Use in production to catch misconfigured hosts early. |
| `false` | Forces CPU regardless of hardware. Useful for reproducibility or cost-constrained deployments. |

### Enabling GPU in Docker Compose

**1. Install the NVIDIA Container Toolkit on the host:**

```bash
# Ubuntu / Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**2. Uncomment the `deploy` block in `docker-compose.yml`** under the `rasa-actions` service:

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**3. Build the actions image with CUDA-enabled torch, set `GPU_ENABLED=true` in your `.env`, and restart:**

The default image installs CPU-only torch, so the `deploy` block alone is not enough.

```bash
# in .env
TORCH_INDEX=https://download.pytorch.org/whl/cu128
GPU_ENABLED=true

docker-compose up --build rasa-actions
```

### CPU-only alternative model

If you cannot use a GPU and need faster CPU inference, you can swap to the xsmall variant (~450 ms) by setting one variable in `.env` and restarting `rasa-actions`:

```bash
CRITICALITY_MODEL=MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33
```

The pipeline API is identical — no other changes required. Note the xsmall accuracy figure above has not yet been measured on the same benchmark as the other rows.

---

## Environment Variables

All variables are documented in `.env.example`. Copy it to `.env` and fill in real values before deploying.

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://chatbot:chatbot@localhost:5432/chatbot` | SQLAlchemy URL for the dashboard (uses `postgres` hostname inside Docker) |
| `DB_HOST` | `localhost` | Postgres host for the Rasa SQL tracker store |
| `DB_PORT` | `5432` | Postgres port |
| `DB_NAME` | `chatbot` | Database name |
| `DB_USER` | `chatbot` | Database user |
| `DB_PASSWORD` | `chatbot` | Database password — **change in production** |
| `JWT_SECRET_KEY` | _(required)_ | Signs dashboard JWT tokens. ≥ 32 random chars; the dashboard won't start without it |
| `DASHBOARD_INGEST_TOKEN` | _(required)_ | Shared secret the action server sends to `/api/update_session`. ≥ 32 random chars |
| `RASA_AUTH_TOKEN` | _(required)_ | Protects the Rasa HTTP API; `docker-compose` refuses to start without it |
| `DASHBOARD_USERNAME` | `admin` | Admin login username |
| `DASHBOARD_PASSWORD` | _(required)_ | Admin login password, ≥ 12 chars, not a default |
| `DASHBOARD_BACKEND_URL` | `http://localhost:5001` | URL the action server uses to push session state |
| `DASHBOARD_PORT` | `5001` | Port the Flask dashboard listens on |
| `RASA_ACTION_ENDPOINT` | `http://localhost:5055/webhook` | Action server webhook URL |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins for the dashboard API |
| `RASA_CORS_ORIGINS` | `*` | Allowed origin for the Rasa chat socket — set to your site in production |
| `TRUSTED_PROXY_COUNT` | `0` | Reverse proxies in front of the dashboard; only then is `X-Forwarded-For` trusted |
| `RATELIMIT_STORAGE_URI` | `memory://` | Rate-limit store; `docker-compose` sets the bundled Redis |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode (never `true` in production) |
| `GUNICORN_WORKERS` | `2` | Dashboard gunicorn worker count |
| `GRAFANA_USER` | `admin` | Grafana admin username |
| `GRAFANA_PASSWORD` | `changeme` | Grafana admin password |
| `GPU_ENABLED` | `auto` | `auto` / `true` / `false` — see [Controlling GPU usage](#controlling-gpu-usage) |
| `TORCH_INDEX` | CPU wheels | Build-time torch index for `rasa-actions`; use `https://download.pytorch.org/whl/cu128` for GPU |
| `PRELOAD_MODELS` | `true` | Load AI models before the action server starts listening, so the first chat doesn't time out |
| `CRITICALITY_MODEL` | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | HuggingFace zero-shot model for criticality scoring |
| `ORDER_PROVIDER` | `mock` | `mock` for development, `rest` for a real backend |
| `ORDER_API_BASE_URL` | _(unset)_ | Required when `ORDER_PROVIDER=rest` |
| `ORDER_API_KEY` | _(unset)_ | Bearer token sent to the order API |
| `ORDER_API_TIMEOUT` | `10` | HTTP timeout (seconds) for order API calls |

---

## AI Components

All three run in the **action server**, not the Rasa NLU pipeline, so only one container needs torch and the model weights. Each user turn is scored once, from `_infer_emotion_slots` in [`actions/actions.py`](actions/actions.py).

### 1. Emotion Detection (`components/emotion.py`)

Uses a DistilBERT model fine-tuned on the GoEmotions dataset to classify each user message into one of six business-friendly emotion buckets: **frustrated**, **anxious**, **disappointed**, **confused**, **neutral**, and **satisfied**. A recency-weighted smoothing window (last 4 turns) prevents a single ambiguous message from flipping the displayed emotion. The raw label and a session-level emotion tally are stored as Rasa slots and forwarded to the dashboard on every turn.

### 2. Criticality & Escalation Scoring (`components/criticality.py`)

Uses `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` (~184 MB) via the HuggingFace zero-shot classification pipeline to score each message as **low**, **medium**, or **high urgency**. The raw model score is adjusted by an emotion prior (e.g. "frustrated" shifts probability mass toward high), then filtered through layered guardrails: YAML keyword overrides, safety/medical/finance regex rules, and structural rules for contrast phrases ("works fine but the battery caught fire"). A decay-weighted conversation-level risk score aggregates per-turn results across the session, with a false-trigger penalty that prevents a single early escalation from holding the conversation at HIGH permanently. Every decision emits a structured reason code for the audit trail.

### 3. Aspect-Based Sentiment Analysis (`components/absa.py`)

Uses MiniLM sentence embeddings to identify which product or service aspect a complaint refers to (e.g. "shipping", "customer support", "product quality") and whether the user's opinion about that aspect is positive, negative, or neutral. Results are rolled up into manager-readable business aspect buckets and pushed to the dashboard as part of the session payload. The dashboard's manager overview aggregates ABSA data across all sessions to surface the most-mentioned negative issues.

---

## Automatically Generated Datasets

The action server appends to two CSV files in `./datasets/` (mounted into the `rasa-actions` container) as conversations happen.

**`conversation_sessions.csv`** — one row per conversation: `Session_ID`, `Start_time`, `User_intent`, `Resolution_Status`, `Final_Sentiment`, `Top_Emotion`, `Emotion_Score`, `Criticality_Level`, `Criticality_Score`, `Conversation_Escalation`, `Criticality_Explanation`, `Handoff_to_agent`, `total_messages`, `conversation_duration`, `Conversation_Feedback`.
Useful for: peak hours for high-risk conversations, escalation frequency and causes, emotion–risk correlation, satisfaction prediction.

**`aspect_sentiment_sessions.csv`** — one row per detected aspect: `Session_ID`, `Original_Text`, `Aspect`, `Aspect_Term`, `Opinion_Term`, `Aspect_Sentiment`, `Confidence`, `Category`, `Business_Aspect`, `Business_Aspect_Display`, `Match_Type`, `Similarity`, `Emotion`, `Emotion_Score`.
Useful for: most-complained-about aspects, aspect-level satisfaction, emotion trends by category, product improvement priorities.

---

## Plugging In a Real Order System

By default the bot uses an in-memory mock with four sample orders. To connect a real backend:

1. Set `ORDER_PROVIDER=rest` in your `.env`.
2. Set `ORDER_API_BASE_URL` to your order API's base URL and `ORDER_API_KEY` if it requires a bearer token.

The REST provider expects three endpoints:

```
GET  {base_url}/orders/{order_id}       → order object
GET  {base_url}/orders?email={email}    → list or {"orders": [...]}
PATCH {base_url}/orders/{order_id}      → body {"address": "..."}
```

Each order object must include: `status`, `eta`, `created_at`, `address`, `email`. See [`integrations/order_provider.py`](integrations/order_provider.py) for the full interface. To integrate a custom backend (Shopify, WooCommerce, SAP), subclass `OrderProvider` and register the new type string in `get_order_provider()`.

---

## License

MIT License. Copyright (c) 2025 Ziad Jreij. See [LICENSE](LICENSE) for details.
