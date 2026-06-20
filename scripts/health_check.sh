#!/usr/bin/env bash
# OllamAGI health check — run after startup to verify all components are working.
set -euo pipefail

PORT=${1:-7654}
BASE="http://localhost:$PORT"
PASS=0
FAIL=0

check() {
  local name="$1" cmd="$2"
  if eval "$cmd" &>/dev/null; then
    echo "  ✓  $name"
    PASS=$((PASS+1))
  else
    echo "  ✗  $name"
    FAIL=$((FAIL+1))
  fi
}

echo ""
echo "OllamAGI Health Check"
echo "─────────────────────"

check "Dashboard reachable"    "curl -sf $BASE/ -o /dev/null"
check "API /api/status"        "curl -sf $BASE/api/status | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d[\"ollama\"][\"ok\"]'"
check "Ollama connected"       "curl -sf $BASE/api/status | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d[\"ollama\"][\"ok\"]'"
check "WebSocket endpoint"     "python3 -c 'import asyncio,websockets; asyncio.run(websockets.connect(\"ws://localhost:$PORT/ws\", open_timeout=2).__aenter__())'"
check "Docker available"       "docker info"
check "Flows API"              "curl -sf $BASE/api/flows -o /dev/null"
check "Tokens API"             "curl -sf $BASE/api/tokens -o /dev/null"
check "Memory search API"      "curl -sf '$BASE/api/memory/search?q=test' -o /dev/null"

echo ""
if [ $FAIL -eq 0 ]; then
  echo "All $PASS checks passed. OllamAGI is healthy."
else
  echo "$PASS passed, $FAIL failed."
  echo "See README.md → Troubleshooting for help."
  exit 1
fi
echo ""
