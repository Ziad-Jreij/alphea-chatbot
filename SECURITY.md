# Security Policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

Report them privately through GitHub: **Security → Report a vulnerability** on this repository (private vulnerability reporting). Include the affected version or commit, steps to reproduce, and the impact you expect.

You should get an acknowledgement within 7 days. Fixes are released as a new version and credited in the changelog unless you prefer to stay anonymous.

## Supported versions

Only the latest release on `main` receives security fixes.

## Deploying Alphea safely

The defaults are safe for a single machine: internal services listen on `127.0.0.1`, and the dashboard and Rasa refuse to start without strong secrets. Before exposing the stack to the internet:

| Setting | Why |
|---|---|
| Run `scripts/setup.sh` or set every secret in `.env` yourself | `JWT_SECRET_KEY`, `DASHBOARD_INGEST_TOKEN` and `RASA_AUTH_TOKEN` must be long random strings. The dashboard rejects short or default values. |
| Never commit `.env` | It is gitignored. Rotate every secret if it leaks. |
| Set `CORS_ORIGINS` and `RASA_CORS_ORIGINS` to your real domain | `*` is for local development only. |
| Put a TLS reverse proxy (nginx, Caddy, Traefik) in front of ports 5001 and 5005 | The services speak plain HTTP. Set `TRUSTED_PROXY_COUNT` to the number of proxies so rate limiting sees real client IPs. |
| Keep Postgres (5432), Prometheus (9090), Grafana (3000) and the action server (5055) off the public internet | They are bound to `127.0.0.1` in `docker-compose.yml`. `/metrics` on the dashboard is unauthenticated by design, for Prometheus. |
| Keep `FLASK_DEBUG=false` | Debug mode allows code execution from the browser. |

## Known dependency alerts

Alphea runs on Rasa Open Source 3.6, which requires Python 3.10. Some dependency fixes are only published for Python 3.11 or later, or need a major upgrade that has to be re-validated against the criticality benchmark. As of v1.0.2 these alerts remain open on purpose:

| Package | Container | Why it is not upgraded | Exposure |
|---|---|---|---|
| `pillow`, `filelock` | rasa-actions | Fixed versions need Python 3.11+. Both are pulled in by the ML libraries. | Alphea does not open images or user-supplied files. |
| `sanic`, `sanic-cors` | rasa-actions | Pinned by `rasa-sdk` 3.6. | The action server is bound to `127.0.0.1` and only Rasa calls it. |
| `transformers`, `sentence-transformers` | rasa-actions | The fixes are in new major versions that change model behaviour. | Only fixed, pinned models from the Hugging Face Hub are loaded; no user-supplied models. |

The internet-facing dashboard (Flask, Flask-Cors, PyJWT, gunicorn) has no open alerts. Moving off Rasa 3.6 and Python 3.10 is the long-term fix; contributions are welcome.

## Data handling

Conversations are stored in PostgreSQL (Rasa tracker store and the `chat_sessions` table) and appended to CSV files in `./datasets/`. They can contain personal data your users type into the chat, such as emails, order numbers and addresses. You are responsible for retention, access control and privacy compliance (for example GDPR) in your deployment.

Criticality scoring and escalation are decision support for human agents. They are not a safety system and can miss urgent messages; see the benchmark section of the README.
