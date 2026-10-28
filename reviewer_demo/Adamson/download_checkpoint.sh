#!/usr/bin/env bash
# Download the published BaiZe Adamson demonstration checkpoint.
# The SHA-256 checksum is pinned to the published GitHub Release asset.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$ROOT/checkpoints/adamson_inference_ema.pth"
URL="https://github.com/simonzqw/BaiZe/releases/download/adamson-smoke-v1/adamson_inference_ema.pth"
EXPECTED_SHA256="3dad6cde7a2530dd4d2736e352b0f2f291ef7cf5e9617bcf0e3380eaa2d2ef89"

verify_sha256() {
    local file="$1"
    printf '%s  %s\n' "$EXPECTED_SHA256" "$file" | sha256sum --check --status
}

mkdir -p "$ROOT/checkpoints"

if [[ -f "$DEST" ]]; then
    if verify_sha256 "$DEST"; then
        echo "PASS: Existing checkpoint matches the published SHA-256."
        echo "Checkpoint: $DEST"
        exit 0
    fi
    echo "ERROR: Existing checkpoint does not match the published SHA-256." >&2
    echo "Refusing to overwrite: $DEST" >&2
    exit 1
fi

TMP="$(mktemp "$DEST.part.XXXXXXXX")"
trap 'rm -f "$TMP"' EXIT

echo "Downloading BaiZe Adamson demonstration checkpoint (approximately 87.29 MiB)..."
if command -v curl >/dev/null 2>&1; then
    curl --fail --location --show-error --retry 3 --connect-timeout 30 \
        --output "$TMP" "$URL"
elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 --timeout=30 --output-document="$TMP" "$URL"
else
    echo "ERROR: curl or wget is required to download the checkpoint." >&2
    exit 1
fi

if ! verify_sha256 "$TMP"; then
    echo "ERROR: Downloaded checkpoint failed SHA-256 verification." >&2
    exit 1
fi

mv "$TMP" "$DEST"
echo "PASS: Checkpoint downloaded and SHA-256 verified."
echo "Checkpoint: $DEST"
