#!/usr/bin/env bash
# One-time setup for a fresh clone:
#   1. creates .env from .env.example with every secret and password generated
#      (an existing .env is kept; only empty required values are filled in)
#   2. creates ./models and ./datasets owned by you, not by root
#   3. downloads the pre-trained Rasa model from the latest GitHub release,
#      so you can skip training (use --no-model to skip this step)
#
# Usage: scripts/setup.sh [--no-model]
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_URL="${ALPHEA_MODEL_URL:-https://github.com/Ziad-Jreij/alphea-chatbot/releases/latest/download/alphea-model.tar.gz}"
FETCH_MODEL=1
for arg in "$@"; do
  case "$arg" in
    --no-model) FETCH_MODEL=0 ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
rand_hex() {  # rand_hex <bytes>
  if command -v openssl >/dev/null 2>&1; then openssl rand -hex "$1"
  else head -c "$1" /dev/urandom | od -An -tx1 | tr -d ' \n'; fi
}
get_env() { grep -E "^$1=" .env | head -n1 | cut -d= -f2- || true; }
set_env() {  # set_env KEY VALUE  (values are hex/URL-safe, no sed escaping needed)
  if grep -qE "^$1=" .env; then sed -i.bak "s|^$1=.*|$1=$2|" .env && rm -f .env.bak
  else printf '%s=%s\n' "$1" "$2" >> .env; fi
}
fill_if_empty() {  # fill_if_empty KEY VALUE
  if [ -z "$(get_env "$1")" ]; then set_env "$1" "$2"; GENERATED+=("$1"); fi
}

command -v docker >/dev/null 2>&1 || { echo "Docker is required: https://docs.docker.com/get-docker/" >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "Docker Compose v2 is required (docker compose ...)." >&2; exit 1; }

GENERATED=()
NEW_ENV=0
if [ ! -f .env ]; then
  cp .env.example .env
  chmod 600 .env
  NEW_ENV=1
  # Only for a brand-new .env: an existing Postgres volume keeps the password
  # it was created with, so changing it later would lock the stack out.
  db_pass="$(rand_hex 16)"
  set_env DB_PASSWORD "$db_pass"
  set_env DATABASE_URL "postgresql://chatbot:${db_pass}@localhost:5432/chatbot"
  GENERATED+=(DB_PASSWORD)
fi

fill_if_empty JWT_SECRET_KEY         "$(rand_hex 32)"
fill_if_empty DASHBOARD_INGEST_TOKEN "$(rand_hex 32)"
fill_if_empty RASA_AUTH_TOKEN        "$(rand_hex 32)"
fill_if_empty DASHBOARD_PASSWORD     "$(rand_hex 12)"
fill_if_empty GRAFANA_PASSWORD       "$(rand_hex 12)"

mkdir -p models datasets

if [ "$FETCH_MODEL" = 1 ]; then
  if ls models/*.tar.gz >/dev/null 2>&1; then
    echo "A trained model is already in ./models; skipping download."
  else
    echo "Downloading the pre-trained model..."
    if curl -fL --progress-bar -o models/alphea-model.tar.gz.part "$MODEL_URL"; then
      mv models/alphea-model.tar.gz.part models/alphea-model.tar.gz
    else
      rm -f models/alphea-model.tar.gz.part
      echo
      echo "No pre-trained model could be downloaded. Train one instead (2-20 min depending on CPU):"
      echo "  docker compose run --rm --no-deps --user \"\$(id -u):0\" rasa train"
    fi
  fi
fi

echo
bold "Setup complete."
[ "$NEW_ENV" = 1 ] && echo "Created .env from .env.example."
if [ "${#GENERATED[@]}" -gt 0 ]; then
  echo "Generated: ${GENERATED[*]}"
else
  echo "All required values in .env were already set; nothing generated."
fi
echo
bold "Dashboard login (stored in .env)"
echo "  user:     $(get_env DASHBOARD_USERNAME)"
echo "  password: $(get_env DASHBOARD_PASSWORD)"
echo
bold "Next"
if ! ls models/*.tar.gz >/dev/null 2>&1; then
  echo "  docker compose run --rm --no-deps --user \"\$(id -u):0\" rasa train"
fi
echo "  docker compose up -d      # first start downloads ~700 MB of AI models"
echo "  open http://localhost:5001/shop.html      (demo shop with the chat assistant)"
echo "  open http://localhost:5001                (agent dashboard)"
