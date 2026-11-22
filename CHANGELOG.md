# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-11-22: first public release

### Added
- `scripts/setup.sh`: one-command setup that generates every secret and password and downloads the pre-trained model.
- Release workflow: pre-built images on GHCR (`ghcr.io/ziad-jreij/alphea-{rasa,actions,dashboard}`) and a trained model attached to each GitHub release.
- Mixed product feedback ("love the design but the battery is awful") has its own rule and response, and runs through ABSA.
- `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`, issue and PR templates, Dependabot.

### Changed
- The dashboard calls its API on the page's own origin, so it works on any host, not just `localhost`.
- Webchat themes 1–3 are served by the dashboard and connect to Rasa on the page's host.
- Grafana requires `GRAFANA_PASSWORD`; there is no `admin/admin` fallback.
- `eval_criticality.py` moved to `scripts/`.

### Removed
- The unused `faq` intent, its rule and its response.
- Rasa Enterprise and unused channel boilerplate from `credentials.yml`.
- JWT decoding errors are no longer returned to API clients.

## 2026-10-25: Escalation recall
- Criticality "risk" strategy: seven risk categories (safety, injury, fraud, account takeover, privacy, medical, legal) scored independently in one batched pass. On the holdout set, HIGH recall rose from 10% to 73.3% and accuracy from 37.8% to 87.8%.
- Benchmark split into dev (244) and holdout (90) cases. Repeated-contact detection. Negation and hearsay handling for finance rules.

## 2026-10-25: Tests and CI
- ~155 unit tests, recorded dialogue test stories, GitHub Actions CI (lint, unit, Rasa validate, train, dialogue tests).
- 334-case criticality benchmark and `eval_criticality.py`.
- Dialogue and guardrail fixes found by the new tests.

## 2026-10-24: Security and reproducible builds
- Services refuse to start with missing or default secrets. Internal ports are bound to `127.0.0.1`. Rasa HTTP API token. Rate limiting backed by Redis.
- A fresh clone builds, trains and runs. Dependencies are pinned in `requirements/`.

## 2026-06-20: Production overhaul
- Docker Compose stack, PostgreSQL tracker store and session table, JWT-protected dashboard, Prometheus and Grafana.
- DeBERTa-v3 zero-shot criticality model, optional GPU support.

## 2025-11 to 2025-12: Thesis version
- Rasa e-commerce support bot with emotion detection (GoEmotions), criticality scoring and escalation, aspect-based sentiment analysis, conversation feedback, an agent dashboard, and two automatically generated analytics datasets.
