# Contributing to Alphea

Thanks for your interest in improving Alphea! Bug reports, benchmark cases, docs fixes and code are all welcome.

## Ground rules

- Be kind. This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
- Report security problems privately. See [SECURITY.md](SECURITY.md), not the issue tracker.
- For anything bigger than a small fix, open an issue first so we can agree on the approach.

## Development setup

You only need Docker and Docker Compose v2. No local Python setup is required.

```bash
git clone https://github.com/Ziad-Jreij/alphea-chatbot.git
cd alphea-chatbot
scripts/setup.sh                       # creates .env with generated secrets
docker compose up -d --build           # builds the images from your working tree
```

The action server bind-mounts `actions/`, `components/`, `integrations/` and `observability/`, so Python changes there only need `docker compose restart rasa-actions`.

Changes to `domain.yml`, `config.yml` or anything under `data/` need a retrain, because Rasa serves the domain baked into the model:

```bash
docker compose run --rm --no-deps --build --user "$(id -u):0" rasa train
docker compose restart rasa
```

## Checks

CI runs these on every pull request, and they must pass:

```bash
# Lint + unit tests (~155 tests, no model downloads)
docker compose run --rm --no-deps -v "$PWD":/app -w /app rasa-actions sh -c \
  "pip install -q -r requirements/dashboard.txt -r requirements/dev.txt && ruff check . --no-cache && pytest -q"

# Rasa data validation + dialogue tests against your latest model
docker compose run --rm --no-deps rasa data validate
docker compose run --rm --no-deps -v "$PWD/tests":/app/tests rasa \
  test core --stories tests/test_stories.yml --fail-on-prediction-errors
```

## Dialogue test stories

Test stories are recorded from real conversations, not written by hand:

1. Add a scenario to [`tests/scenarios.yml`](tests/scenarios.yml).
2. With the stack running, run `python3 scripts/record_test_stories.py`. It prints each transcript and rewrites `tests/test_stories.yml`.
3. **Read the transcripts before committing.** A recorded story asserts that the current replies are correct.

## Criticality benchmark

[`data/eval/criticality_benchmark.yml`](data/eval/criticality_benchmark.yml) has a **dev** split for tuning and a **holdout** split for reporting only. When you change criticality logic:

- tune on `--split dev`;
- don't tune on the holdout split. Report it once, in the PR;
- new cases go in `dev` unless they were written before the change they evaluate.

```bash
docker compose run --rm --no-deps -v "$PWD":/app -w /app rasa-actions \
  python scripts/eval_criticality.py --split dev --show-errors
```

## Pull requests

- Keep each PR focused on one change, and include tests for behaviour changes.
- Describe what changed and how you verified it (unit tests, dialogue tests, benchmark numbers, a screenshot for UI changes).
- Dependencies are pinned in [`requirements/`](requirements/). Update the pin and explain why.
- Code style: `ruff check .` must pass. Match the surrounding code otherwise.

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
