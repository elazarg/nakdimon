#!/bin/sh
# Build the Nakdimon web demo: compile packages/nakdimon-js if needed, then
# copy its dist/ (as web/lib/) and the ONNX model into web/.
set -eu

# cd to the directory this script lives in (web/), so it can be run from anywhere.
cd "$(dirname "$0")"

PKG_DIR=../packages/nakdimon-js
DIST_DIR="$PKG_DIR/dist"

# --- 1. Build packages/nakdimon-js if dist/ is missing or stale ---------
DIST_MARKER="$DIST_DIR/index.js"

need_build=0
if [ ! -f "$DIST_MARKER" ]; then
  need_build=1
else
  # Stale if any src/ file is newer than the marker dist/ file. (Comparing
  # against the dist/ *directory*'s mtime is unreliable: on Linux, a
  # directory's mtime only changes when entries are added/removed, not when
  # tsc overwrites an existing file's contents in place.)
  newest_src=$(find "$PKG_DIR/src" -type f -newer "$DIST_MARKER" 2>/dev/null | head -n1)
  if [ -n "$newest_src" ]; then
    need_build=1
  fi
fi

if [ "$need_build" -eq 1 ]; then
  echo "==> packages/nakdimon-js/dist is missing or stale; building..."
  ( cd "$PKG_DIR" && npm install && npm run build )
  echo "==> build complete."
else
  echo "==> packages/nakdimon-js/dist is up to date; skipping build."
fi

# --- 2. Copy the built library into web/lib/ -----------------------------
rm -rf lib
mkdir -p lib
cp -R "$DIST_DIR"/. lib/
echo "==> copied $DIST_DIR -> web/lib/"

# --- 3. Copy the ONNX model (and sidecar card, if present) ---------------
MODEL_NEW=../packages/nakdimon/src/nakdimon/data/Nakdimon.onnx
MODEL_OLD=../nakdimon/data/Nakdimon.onnx

if [ -f "$MODEL_NEW" ]; then
  MODEL_SRC="$MODEL_NEW"
elif [ -f "$MODEL_OLD" ]; then
  MODEL_SRC="$MODEL_OLD"
else
  echo "==> ERROR: could not find Nakdimon.onnx in either:" >&2
  echo "      $MODEL_NEW" >&2
  echo "      $MODEL_OLD" >&2
  exit 1
fi

cp "$MODEL_SRC" Nakdimon.onnx
echo "==> copied $MODEL_SRC -> web/Nakdimon.onnx"

CARD_SRC="${MODEL_SRC}.json"
if [ -f "$CARD_SRC" ]; then
  cp "$CARD_SRC" Nakdimon.onnx.json
  echo "==> copied $CARD_SRC -> web/Nakdimon.onnx.json"
else
  echo "==> no sidecar model card at $CARD_SRC; using built-in default card (v1)."
fi

echo "==> done."
