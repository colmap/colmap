#!/usr/bin/env bash
set -euo pipefail

# ── CodeArtifact configuration ─────────────────────────────────────────────
DOMAIN=""           # e.g. my-domain
DOMAIN_OWNER=""     # AWS account ID (12 digits)
REPOSITORY="pycolmap"       # e.g. my-repo
REGION=""           # e.g. us-east-1
# ──────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<EOF
Usage: $0 <wheel_path>

Upload a Python wheel to AWS CodeArtifact.

Arguments:
  wheel_path   Path to the .whl file (absolute or relative to this script)

Examples:
  $0 dist/pycolmap-4.1.0.dev0-cp312-cp312-linux_x86_64-cuda13.0.whl
EOF
  exit 1
}

# ── Validate arguments ─────────────────────────────────────────────────────
[[ $# -ne 1 ]] && usage

WHEEL_PATH="$1"
# Resolve relative paths against the caller's working directory
[[ "$WHEEL_PATH" != /* ]] && WHEEL_PATH="$PWD/$WHEEL_PATH"

[[ ! -f "$WHEEL_PATH" ]] && { echo "Error: wheel not found: $WHEEL_PATH"; exit 1; }

# ── Validate configuration ─────────────────────────────────────────────────
for var in DOMAIN DOMAIN_OWNER REPOSITORY REGION; do
  [[ -z "${!var}" ]] && { echo "Error: \$$var is not set. Edit the configuration block at the top of this script."; exit 1; }
done

# ── Check dependencies ─────────────────────────────────────────────────────
if ! command -v twine &>/dev/null; then
  echo "twine not found — installing..."
  pip install --quiet twine
fi

if ! command -v aws &>/dev/null; then
  echo "Error: aws CLI is not installed."
  exit 1
fi

# ── Authenticate with CodeArtifact ─────────────────────────────────────────
echo "Authenticating with CodeArtifact (domain: $DOMAIN, repo: $REPOSITORY)..."
aws codeartifact login \
  --tool twine \
  --domain "$DOMAIN" \
  --domain-owner "$DOMAIN_OWNER" \
  --repository "$REPOSITORY" \
  --region "$REGION"

# ── Upload ─────────────────────────────────────────────────────────────────
echo "Uploading: $(basename "$WHEEL_PATH")"
twine upload --repository codeartifact "$WHEEL_PATH"

echo "Done."
