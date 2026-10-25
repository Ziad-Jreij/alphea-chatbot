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

`eval_criticality.py` scores models on [`data/eval/criticality_benchmark.yml`](data/eval/criticality_benchmark.yml): 334 labelled messages split into **dev** (244, used for design and tuning) and **holdout** (90, written before the current design and only scored for reporting). Both splits are balanced across low/medium/high and include negation, hearsay, contrast and "safety word" traps. Two modes:

- **raw** — the model alone
- **pipeline** — the production path: keyword guardrails → model → emotion prior → structural rules

Holdout results, production pipeline (2026-10, CPU):

| Model | Strategy | Accuracy | Macro-F1 | HIGH precision | HIGH recall | LOW→HIGH false alarms | Median latency |
|---|---|---|---|---|---|---|---|
| **DeBERTa-v3-base** | **risk** (default) | **87.8%** | **87.5%** | **88.0%** | **73.3%** | 1 / 30 | 195 ms |
| DeBERTa-v3-xsmall | risk | 78.9% | 78.8% | 80.0% | 66.7% | 3 / 30 | 56 ms |
| DeBERTa-v3-base | urgency (original) | 37.8% | 33.4% | 50.0% | 10.0% | 1 / 30 | 107 ms |

**Why the risk strategy:** the original design asked the model to rank "low / medium / high urgency", so a fire or a hacked account had to out-score every other reading of the message, and mostly didn't (10% HIGH recall). The risk strategy scores seven risk categories (safety hazard, injury, fraud, account takeover, privacy breach, medical risk, legal threat) **independently**. Any one of them above 0.70 escalates, and problem vs. routine categories split the rest. All 15 categories run in one batched forward pass.

Known misses on the holdout include a broken carbon-monoxide detector, a gas smell after an installation, a phishing request for a card PIN, and a chargeback threat. Treat escalation as decision support for agents, not a safety system.

Caveats: the holdout set was scored a second time after one category was reworded in response to a *live-testing* failure (plain return requests were being escalated), not a holdout error. 290 of the 334 labels were drafted in October 2026 and are marked `source: draft` pending human review. On the original 40-message thesis set, the original urgency strategy scores 60% raw, matching the thesis result.

Reproduce (dev for tuning, holdout for reporting):

```bash
docker compose run --rm --no-deps -v "$PWD":/app -w /app rasa-actions \
  python eval_criticality.py --strategies risk urgency --split holdout --show-errors
```

### CPU vs GPU latency

| Hardware | Median latency / message | Recommended for |
|---|---|---|
| CPU only | ~200 ms criticality per message measured on a 24-core laptop CPU with 8 torch threads; expect more on small cloud instances | Development, low-traffic demos |
| NVIDIA GPU (e.g. T4) | ~30–50 ms | Production, > 10 concurrent users |

**CPU is usable** for development and low-traffic demos: each user turn is scored once by the action server (emotion + criticality + ABSA), adding roughly 0.1–1 s to the reply depending on the CPU. Measure on your target hardware with `eval_criticality.py` before putting real traffic on it.

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

`MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33` is ~3× smaller and ~3.5× faster (56 ms vs 195 ms per message on CPU), at a cost of about 9 points of accuracy and 7 points of HIGH recall on the holdout set (see the benchmark above). Switch by setting one variable in `.env` and restarting `rasa-actions`:

```bash
CRITICALITY_MODEL=MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33
```

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
| `CRITICALITY_STRATEGY` | `risk` | `risk` scores safety/fraud/privacy/medical/legal categories independently; `urgency` is the original low/medium/high softmax |
| `TORCH_NUM_THREADS` | min(8, cores) | CPU threads for the models; more than ~8 increases tail latency |
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

Scores each message as **low**, **medium** or **high** criticality with `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` (~184 MB). By default (`CRITICALITY_STRATEGY=risk`) the model scores 15 descriptive categories independently: seven risk categories (fire/shock/overheating, injury, fraud, account takeover, privacy breach, medical risk, legal threat) and eight problem or routine categories. HIGH means any risk category clears 0.70. An emotion prior (e.g. "frustrated", "anxious") shifts the scores in log-odds space. Layered guardrails sit around the model:
- YAML high keywords decide immediately; medium keywords set a floor.
- Regex rules for safety, medical, finance, contrast ("works fine but the battery caught fire") and repeated unresolved contact. The finance rules ignore negation and hearsay ("there was no unauthorized charge", "I heard it's a scam").

A decay-weighted conversation-level risk score aggregates the turns by level, with a false-trigger penalty so one early escalation doesn't hold a calm conversation at HIGH. Every decision emits a reason code and a one-sentence explanation, e.g. *"Escalated to HIGH because the model detected a safety hazard … (score 0.93)."*

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

## Development & Testing

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on every push and pull request:

| Job | What it checks |
|---|---|
| Lint + unit tests | `ruff check .` and `pytest` (~155 tests, no model downloads): criticality guardrails, emotion bucketing, ABSA pairing, validators, order providers, dashboard auth/ingest/rate limiting, form and conversation-risk logic |
| Rasa dialogue tests | `rasa data validate`, a full `rasa train`, then `rasa test core` on [`tests/test_stories.yml`](tests/test_stories.yml) with `--fail-on-prediction-errors` |

Run the same checks locally with Docker (no local Python setup needed):

```bash
# Unit tests + lint, in the action-server image
docker compose run --rm --no-deps -v "$PWD":/app -w /app rasa-actions sh -c \
  "pip install -q -r requirements/dashboard.txt -r requirements/dev.txt && ruff check . --no-cache && pytest -q"

# Dialogue tests against the latest trained model
docker compose run --rm --no-deps -v "$PWD/tests":/app/tests rasa \
  test core --stories tests/test_stories.yml --fail-on-prediction-errors
```

Dependencies are pinned in [`requirements/`](requirements/) and shared by the Dockerfiles and CI.

Test stories are recorded from real conversations, not written by hand. Add a scenario to [`tests/scenarios.yml`](tests/scenarios.yml), then, with the stack running:

```bash
python3 scripts/record_test_stories.py   # prints every transcript, rewrites tests/test_stories.yml
```

Read the transcripts before committing: a recorded story asserts that the current replies are correct.

---

## License

MIT License. Copyright (c) 2025 Ziad Jreij. See [LICENSE](LICENSE) for details.
