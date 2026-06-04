#!/usr/bin/env bash
# Quick smoke-test for the Whisper STT API.
# Usage: ./test_api.sh [audio_file]

API=http://localhost:8000
FILE=${1:-""}

echo "=== Health check ==="
curl -s "$API/health" | python3 -m json.tool

if [[ -n "$FILE" ]]; then
  echo ""
  echo "=== Transcribe: $FILE ==="
  curl -s -X POST "$API/transcribe" \
    -F "file=@$FILE" \
    -F "task=transcribe" \
    -F "word_timestamps=false" | python3 -m json.tool
fi
