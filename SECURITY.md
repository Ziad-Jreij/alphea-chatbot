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

## Data handling

Conversations are stored in PostgreSQL (Rasa tracker store and the `chat_sessions` table) and appended to CSV files in `./datasets/`. They can contain personal data your users type into the chat, such as emails, order numbers and addresses. You are responsible for retention, access control and privacy compliance (for example GDPR) in your deployment.

Criticality scoring and escalation are decision support for human agents. They are not a safety system and can miss urgent messages; see the benchmark section of the README.
