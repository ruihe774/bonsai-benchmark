#!/bin/bash
# Adjust the wall-clock deadline of a benchmark run while it is in progress.
#
#   sandbox/deadline.sh RUN_NAME show
#   sandbox/deadline.sh RUN_NAME set EPOCH
#   sandbox/deadline.sh RUN_NAME extend DURATION
#   sandbox/deadline.sh RUN_NAME shorten DURATION
#
# RUNS_ROOT selects the runs directory (default /mnt/workspace/bonsai-runs).
# DURATION is a number of seconds or a value with an s/m/h suffix (30m, 2h).
#
# The runner polls the live deadline once a second, so a change takes effect
# without restarting the container. The control directory is mounted read-only
# in the agent container, so only the host operator can adjust it. Adjustment
# is deliberately manual: the runner does not watch upstream rate limits.
set -euo pipefail

ROOT="${RUNS_ROOT:-/mnt/workspace/bonsai-runs}"
NAME="${1:?usage: deadline.sh RUN_NAME show | set EPOCH | extend DURATION | shorten DURATION}"
ACTION="${2:?usage: deadline.sh RUN_NAME show | set EPOCH | extend DURATION | shorten DURATION}"
CTL="$ROOT/$NAME.control"
FILE="$CTL/deadline"
INITIAL_FILE="$CTL/initial-deadline"
LOG="$CTL/adjustments.log"

[ -f "$FILE" ] || {
  echo "no live deadline for '$NAME' ($FILE not found)" >&2
  echo "start the run with sandbox/run_agent.sh first" >&2
  exit 1
}

read_deadline() {
  local d
  d=$(tr -d '[:space:]' < "$1")
  case "$d" in
    ''|*[!0-9]*) return 1 ;;
  esac
  printf '%s\n' "$((10#$d))"
}

duration_s() {
  local spec=$1 n mul
  case "$spec" in
    *s) n=${spec%s}; mul=1 ;;
    *m) n=${spec%m}; mul=60 ;;
    *h) n=${spec%h}; mul=3600 ;;
    *)  n=$spec;     mul=1 ;;
  esac
  case "$n" in
    ''|*[!0-9]*) echo "invalid duration '$spec'" >&2; return 1 ;;
  esac
  n=$((10#$n))
  [ "$n" -gt 0 ] || { echo "duration must be positive" >&2; return 1; }
  printf '%s\n' "$((n * mul))"
}

fmt_epoch() {
  date -d "@$1" '+%Y-%m-%d %H:%M:%S %Z'
}

show() {
  local cur now initial left
  cur=$(read_deadline "$FILE") || {
    echo "unreadable deadline in $FILE" >&2
    exit 1
  }
  now=$(date +%s)
  printf 'run:       %s\n' "$NAME"
  printf 'deadline:  %s (epoch %d)\n' "$(fmt_epoch "$cur")" "$cur"
  if [ "$cur" -gt "$now" ]; then
    left=$((cur - now))
    printf 'remaining: %dh %02dm %02ds\n' \
      $((left / 3600)) $((left % 3600 / 60)) $((left % 60))
  else
    printf 'remaining: 0 (deadline passed %ds ago)\n' "$((now - cur))"
  fi
  if [ -f "$INITIAL_FILE" ]; then
    initial=$(read_deadline "$INITIAL_FILE" || true)
    if [ -n "$initial" ]; then
      printf 'initial:   %s (epoch %d)\n' "$(fmt_epoch "$initial")" "$initial"
    fi
  fi
  if [ -s "$LOG" ]; then
    printf 'adjustments:\n'
    while read -r ts old new what arg; do
      printf '  %s  %d -> %d  %s %s\n' \
        "$(fmt_epoch "$ts")" "$old" "$new" "$what" "$arg"
    done < "$LOG"
  fi
}

adjust() {
  local action=$1 arg=$2 cur new now delta tmp
  cur=$(read_deadline "$FILE") || {
    echo "unreadable deadline in $FILE" >&2
    exit 1
  }
  case "$action" in
    set)
      case "$arg" in
        ''|*[!0-9]*) echo "set needs an absolute Unix epoch" >&2; exit 1 ;;
      esac
      new=$((10#$arg))
      ;;
    extend)
      delta=$(duration_s "$arg") || exit 1
      new=$((cur + delta))
      ;;
    shorten)
      delta=$(duration_s "$arg") || exit 1
      new=$((cur - delta))
      ;;
  esac
  [ "$new" -gt 0 ] || { echo "deadline would be <= 0" >&2; exit 1; }

  # Replace the file atomically; the runner re-reads it every second.
  tmp=$(mktemp "$CTL/.deadline.XXXXXX")
  printf '%d\n' "$new" > "$tmp"
  mv -f "$tmp" "$FILE"
  now=$(date +%s)
  printf '%d %d %d %s %s\n' "$now" "$cur" "$new" "$action" "$arg" >> "$LOG"

  printf 'deadline: %d -> %d (%+d s)\n' "$cur" "$new" "$((new - cur))"
  if [ "$new" -le "$now" ]; then
    echo "warning: deadline is in the past; the runner will stop the agent now" >&2
  fi
}

case "$ACTION" in
  show)    show ;;
  set)     adjust set "${3:?set needs EPOCH}" ;;
  extend)  adjust extend "${3:?extend needs DURATION}" ;;
  shorten) adjust shorten "${3:?shorten needs DURATION}" ;;
  *)
    echo "unknown action '$ACTION' (use show, set, extend or shorten)" >&2
    exit 2
    ;;
esac
