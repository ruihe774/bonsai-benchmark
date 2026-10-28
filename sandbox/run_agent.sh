#!/bin/bash
# Run Claude Code inside the podman sandbox for one benchmark run.
#
#   ANTHROPIC_BASE_URL=https://api.example ANTHROPIC_AUTH_TOKEN=... \
#     sandbox/run_agent.sh RUN_NAME [RUNS_ROOT]
#
# The agent container has no internet route. It can only reach the proxy
# container on an internal podman network; the proxy holds the credential and
# forwards only the Messages API paths. The wall-clock deadline can be changed
# while the run is in progress with sandbox/deadline.sh.
set -euo pipefail

BENCH="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
NAME="${1:?usage: run_agent.sh RUN_NAME [RUNS_ROOT]}"
ROOT="${2:-/mnt/workspace/bonsai-runs}"
RUN="$ROOT/$NAME"
SETTINGS="$BENCH/sandbox/generated/$NAME.settings.json"

[ -d "$RUN" ] || { echo "missing run dir $RUN; run sandbox/prepare_run.sh first" >&2; exit 1; }
[ -f "$SETTINGS" ] || { echo "missing $SETTINGS" >&2; exit 1; }
: "${ANTHROPIC_BASE_URL:?set ANTHROPIC_BASE_URL (upstream API endpoint)}"
: "${ANTHROPIC_AUTH_TOKEN:?set ANTHROPIC_AUTH_TOKEN (upstream credential)}"

AGENT_IMAGE="${AGENT_IMAGE:-bonsai-agent:1.0}"
PROXY_IMAGE="${PROXY_IMAGE:-bonsai-proxy:1.0}"
INT_NET="${INT_NET:-bonsai-int}"
EGRESS_NET="${EGRESS_NET:-bonsai-egress}"
PROXY_IP="${PROXY_IP:-10.89.0.2}"
AGENT_IP="${AGENT_IP:-10.89.0.3}"
PNAME="bonsai-proxy-$NAME"
ANAME="bonsai-agent-$NAME"
RUN_TIMEOUT="${RUN_TIMEOUT:-7200}"    # initial wall-clock budget, seconds (default 2 h)
START_EPOCH=$(date +%s)
DEADLINE_EPOCH=$(( START_EPOCH + RUN_TIMEOUT ))
# Operator-adjustable live deadline. sandbox/deadline.sh rewrites
# $CTL/deadline; the runner polls it and stops the container at that time. The
# directory is mounted read-only in the container, so the agent cannot extend
# its own session.
CTL="$ROOT/$NAME.control"
WATCH_PID=""

live_deadline() {
  local d
  d=$(tr -d '[:space:]' < "$CTL/deadline" 2>/dev/null || true)
  case "$d" in ''|*[!0-9]*) d=$DEADLINE_EPOCH ;; esac
  printf '%s\n' "$d"
}

watch_deadline() {
  local pid=$1 dl
  while kill -0 "$pid" 2>/dev/null; do
    dl=$(live_deadline)
    if [ "$(date +%s)" -ge "$dl" ]; then
      printf '%d\n' "$dl" > "$CTL/terminated"
      # A paused ("rate-limited") container stays frozen; thaw it so the stop
      # signals are delivered. Pausing/unpausing is itself left to the operator.
      podman unpause "$ANAME" >/dev/null 2>&1 || true
      podman stop --time 60 "$ANAME" >/dev/null 2>&1 || true
      # The container may not exist yet (very short deadline), or stop may have
      # failed; also terminate the podman client like the old timeout did.
      if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
        sleep 5
        kill -KILL "$pid" 2>/dev/null || true
      fi
      return
    fi
    sleep 1
  done
}

# Optional model pin for benchmarking several models through the same proxy.
MODEL_ENV=()
if [ -n "${BENCH_MODEL:-}" ]; then
  for v in ANTHROPIC_MODEL ANTHROPIC_DEFAULT_OPUS_MODEL \
           ANTHROPIC_DEFAULT_SONNET_MODEL ANTHROPIC_DEFAULT_HAIKU_MODEL \
           ANTHROPIC_DEFAULT_FABLE_MODEL; do
    MODEL_ENV+=(-e "$v=$BENCH_MODEL")
  done
fi

# Claude Code tuning. Defaults bound credit burn (no fan-out of subagents);
# model-dependent knobs (effort, compact window, context/output size, turn
# cap) are set only when the operator asks for them. See README.md.
TUNING_ENV=(
  -e "CLAUDE_CODE_SUBAGENT_MODEL=${BENCH_SUBAGENT_MODEL:-inherit}"
  -e "CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS=${BENCH_MAX_SUBAGENTS:-1}"
  -e "CLAUDE_CODE_MAX_SUBAGENTS_PER_SESSION=${BENCH_MAX_SUBAGENTS_PER_SESSION:-8}"
  -e "CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH=${BENCH_MAX_SUBAGENT_DEPTH:-1}"
  -e "CLAUDE_CODE_DISABLE_EXPLORE_PLAN_AGENTS=${BENCH_DISABLE_EXPLORE_PLAN_AGENTS:-1}"
  -e "CLAUDE_CODE_ENABLE_TODO_TOOLS=${BENCH_ENABLE_TODO_TOOLS:-1}"
  -e "CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1"
  -e "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL=1"
  -e "CLAUDE_CODE_ENABLE_AWAY_SUMMARY=0"
  -e "CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION=0"
  -e "CLAUDE_CODE_DISABLE_CFC_PROMPT=1"
  -e "CLAUDE_CODE_DISABLE_AUTO_MEMORY=1"
  -e "CLAUDE_CODE_DISABLE_CRON=1"
)
if [ -n "${BENCH_EFFORT:-}" ]; then
  TUNING_ENV+=(-e "CLAUDE_CODE_EFFORT_LEVEL=$BENCH_EFFORT")
fi
if [ -n "${BENCH_COMPACT_WINDOW:-}" ]; then
  TUNING_ENV+=(-e "CLAUDE_CODE_AUTO_COMPACT_WINDOW=$BENCH_COMPACT_WINDOW")
fi
if [ -n "${BENCH_MAX_CONTEXT:-}" ]; then
  TUNING_ENV+=(-e "CLAUDE_CODE_MAX_CONTEXT_TOKENS=$BENCH_MAX_CONTEXT")
fi
if [ -n "${BENCH_MAX_OUTPUT:-}" ]; then
  TUNING_ENV+=(-e "CLAUDE_CODE_MAX_OUTPUT_TOKENS=$BENCH_MAX_OUTPUT")
fi
if [ -n "${BENCH_MAX_TURNS:-}" ]; then
  TUNING_ENV+=(-e "CLAUDE_CODE_MAX_TURNS=$BENCH_MAX_TURNS")
fi
if [ -n "${BENCH_SIMPLE_SYSTEM_PROMPT:-}" ]; then
  TUNING_ENV+=(-e "CLAUDE_CODE_SIMPLE_SYSTEM_PROMPT=$BENCH_SIMPLE_SYSTEM_PROMPT")
fi

# Record the effective configuration next to the run for later analysis.
{
  printf 'BENCH_MODEL=%s\n' "${BENCH_MODEL:-}"
  printf 'CLAUDE_CODE_EFFORT_LEVEL=%s\n' "${BENCH_EFFORT:-}"
  printf 'CLAUDE_CODE_AUTO_COMPACT_WINDOW=%s\n' "${BENCH_COMPACT_WINDOW:-}"
  printf 'CLAUDE_CODE_MAX_CONTEXT_TOKENS=%s\n' "${BENCH_MAX_CONTEXT:-}"
  printf 'CLAUDE_CODE_MAX_OUTPUT_TOKENS=%s\n' "${BENCH_MAX_OUTPUT:-}"
  printf 'CLAUDE_CODE_MAX_TURNS=%s\n' "${BENCH_MAX_TURNS:-}"
  printf 'CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS=%s\n' "${BENCH_MAX_SUBAGENTS:-1}"
  printf 'CLAUDE_CODE_MAX_SUBAGENTS_PER_SESSION=%s\n' "${BENCH_MAX_SUBAGENTS_PER_SESSION:-8}"
  printf 'CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH=%s\n' "${BENCH_MAX_SUBAGENT_DEPTH:-1}"
  printf 'CLAUDE_CODE_DISABLE_EXPLORE_PLAN_AGENTS=%s\n' "${BENCH_DISABLE_EXPLORE_PLAN_AGENTS:-1}"
  printf 'CLAUDE_CODE_SUBAGENT_MODEL=%s\n' "${BENCH_SUBAGENT_MODEL:-inherit}"
  printf 'CLAUDE_CODE_ENABLE_TODO_TOOLS=%s\n' "${BENCH_ENABLE_TODO_TOOLS:-1}"
  printf 'CLAUDE_CODE_SIMPLE_SYSTEM_PROMPT=%s\n' "${BENCH_SIMPLE_SYSTEM_PROMPT:-}"
} > "$ROOT/$NAME.tuning"

podman image exists "$AGENT_IMAGE" || { echo "missing image $AGENT_IMAGE; run setup.sh" >&2; exit 1; }
podman image exists "$PROXY_IMAGE" || { echo "missing image $PROXY_IMAGE; run setup.sh" >&2; exit 1; }

if ! podman network exists "$INT_NET"; then
  podman network create --internal --subnet 10.89.0.0/24 "$INT_NET" >/dev/null
fi
if ! podman network exists "$EGRESS_NET"; then
  podman network create "$EGRESS_NET" >/dev/null
fi

rm -rf "$CTL"
mkdir -p "$CTL"
printf '%d\n' "$DEADLINE_EPOCH" > "$CTL/deadline"
printf '%d\n' "$DEADLINE_EPOCH" > "$CTL/initial-deadline"
: > "$CTL/adjustments.log"

cleanup() {
  if [ -n "$WATCH_PID" ]; then kill "$WATCH_PID" >/dev/null 2>&1 || true; fi
  podman logs "$PNAME" >"$ROOT/$NAME.proxy.jsonl" 2>/dev/null || true
  podman rm -f "$PNAME" >/dev/null 2>&1 || true
  podman rm -f "$ANAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

podman rm -f "$PNAME" "$ANAME" >/dev/null 2>&1 || true
podman run -d --name "$PNAME" \
  --network "$INT_NET:ip=$PROXY_IP" --network "$EGRESS_NET" \
  -e HOST="$PROXY_IP" -e PORT=8787 \
  -e ANTHROPIC_BASE_URL="$ANTHROPIC_BASE_URL" \
  -e ANTHROPIC_AUTH_TOKEN="$ANTHROPIC_AUTH_TOKEN" \
  -e ALLOWED_PATHS="${ALLOWED_PATHS:-/v1/messages,/v1/messages/count_tokens,/v1/models}" \
  -e ALLOWED_METHODS="${ALLOWED_METHODS:-POST,GET}" \
  -e AUTH_HEADER="${AUTH_HEADER:-authorization}" \
  -e MAX_REQUESTS="${MAX_REQUESTS:-10000}" \
  -e LOG_BODIES="${LOG_BODIES:-0}" \
  -v "$BENCH/sandbox/proxy:/proxy:ro" \
  "$PROXY_IMAGE" >/dev/null

for _ in $(seq 1 100); do
  if podman logs "$PNAME" 2>&1 | grep -q "listening on"; then break; fi
  sleep 0.2
done
if ! podman logs "$PNAME" 2>&1 | grep -q "listening on"; then
  echo "proxy failed to start; logs:" >&2
  podman logs "$PNAME" >&2 || true
  exit 1
fi

set +e
podman run --rm --name "$ANAME" \
  --device /dev/dri --group-add keep-groups --security-opt label=disable \
  --network "$INT_NET:ip=$AGENT_IP" \
  --shm-size=4g --tmpfs /tmp:rw,size=8g \
  -v "$CTL:/run/bench:ro" \
  -e BENCH_DEADLINE_FILE=/run/bench/deadline \
  -e BENCH_DEADLINE_EPOCH="$DEADLINE_EPOCH" \
  -v "$BENCH/sandbox/time-left.sh:/usr/local/bin/time-left:ro" \
  -v "$RUN:/workspace" \
  -v "$BENCH/deps/docs:/workspace/docs:ro" \
  -v "$BENCH/deps/data:/workspace/data:ro" \
  -v "$BENCH/deps/ref:/workspace/ref:ro" \
  -v "$SETTINGS:/run/claude-settings.json:ro" \
  -v "$RUN/.claude-home:/root/.claude" \
  -v "$RUN/.claude.json:/root/.claude.json" \
  -e HOME=/root \
  -e CLAUDE_CONFIG_DIR=/root/.claude \
  "${MODEL_ENV[@]}" \
  "${TUNING_ENV[@]}" \
  -e ANTHROPIC_BASE_URL="http://$PROXY_IP:8787" \
  -e ANTHROPIC_AUTH_TOKEN="benchmark-dummy-token" \
  -e CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
  -e DISABLE_TELEMETRY=1 \
  "$AGENT_IMAGE" \
  bash -lc 'cd /workspace && claude -p "$(cat PROMPT.md)" --settings /run/claude-settings.json --setting-sources project --safe-mode --strict-mcp-config --permission-mode dontAsk --output-format stream-json --verbose' \
  >"$ROOT/$NAME.transcript.jsonl" 2>"$ROOT/$NAME.stderr.log" &
AGENT_PID=$!
watch_deadline "$AGENT_PID" &
WATCH_PID=$!
wait "$AGENT_PID"
rc=$?
kill "$WATCH_PID" >/dev/null 2>&1 || true
wait "$WATCH_PID" >/dev/null 2>&1 || true
WATCH_PID=""
set -e
END_EPOCH=$(date +%s)
timed_out=false
[ -f "$CTL/terminated" ] && timed_out=true
FINAL_DEADLINE_EPOCH=$(live_deadline)
ADJUSTMENTS=$(grep -c . "$CTL/adjustments.log" 2>/dev/null || true)
case "$ADJUSTMENTS" in ''|*[!0-9]*) ADJUSTMENTS=0 ;; esac
deadline_adjusted=false
[ "$FINAL_DEADLINE_EPOCH" -ne "$DEADLINE_EPOCH" ] && deadline_adjusted=true
printf '{"start_epoch":%d,"end_epoch":%d,"budget_s":%d,"effective_budget_s":%d,"elapsed_s":%d,"exit":%d,"timed_out":%s,"deadline_epoch":%d,"initial_deadline_epoch":%d,"deadline_adjusted":%s,"deadline_adjustments":%d}\n' \
  "$START_EPOCH" "$END_EPOCH" "$RUN_TIMEOUT" "$((FINAL_DEADLINE_EPOCH - START_EPOCH))" \
  "$((END_EPOCH - START_EPOCH))" "$rc" "$timed_out" "$FINAL_DEADLINE_EPOCH" \
  "$DEADLINE_EPOCH" "$deadline_adjusted" "$ADJUSTMENTS" > "$ROOT/$NAME.meta.json"
echo "agent exit: $rc (elapsed $((END_EPOCH - START_EPOCH))s; deadline $((FINAL_DEADLINE_EPOCH - START_EPOCH))s after start)"
exit "$rc"
