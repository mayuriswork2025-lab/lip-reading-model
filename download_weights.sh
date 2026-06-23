#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
MODELDIR="$ROOT/evaluation/models"
mkdir -p "$MODELDIR"
URL="$1"
OUT="$2"
if [ -z "$URL" ]; then
  echo "Usage: $0 <weights_url> [output_filename]"
  exit 1
fi
FNAME="${OUT:-$(basename "$URL") }"
curl -L -o "$MODELDIR/$FNAME" "$URL"
if [ $? -ne 0 ]; then
  echo "Download failed"
  exit 2
fi

echo "Downloaded to $MODELDIR/$FNAME"
