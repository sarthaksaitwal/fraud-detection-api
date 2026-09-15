#!/usr/bin/env bash
# Fetch the Kaggle credit-card fraud dataset into data/raw/.
#
# Requires the Kaggle API token at ~/.kaggle/kaggle.json
#   1. https://www.kaggle.com/settings/account -> API -> "Create New Token"
#   2. move the downloaded kaggle.json to ~/.kaggle/kaggle.json
#   3. pip install kaggle
#
# Manual alternative (no CLI): download from
#   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# and unzip creditcard.csv into data/raw/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/raw"
TARGET="$DEST/creditcard.csv"

if [ -f "$TARGET" ]; then
  echo "already present: $TARGET"
  exit 0
fi

if ! command -v kaggle >/dev/null 2>&1; then
  echo "ERROR: the 'kaggle' CLI is not installed (pip install kaggle)." >&2
  echo "Or download manually -- see the header of this script." >&2
  exit 1
fi

mkdir -p "$DEST"
kaggle datasets download -d mlg-ulb/creditcardfraud -p "$DEST" --unzip
echo "downloaded -> $TARGET"
