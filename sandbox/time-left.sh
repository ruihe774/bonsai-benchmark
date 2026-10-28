#!/bin/sh
# Print the wall-clock time remaining in the benchmark run.
#
# The runner keeps the live deadline in BENCH_DEADLINE_FILE and updates it when
# the operator adjusts the budget. BENCH_DEADLINE_EPOCH is the launch value and
# is used as a fallback for runs started without a control file.
deadline="${BENCH_DEADLINE_EPOCH:-}"
if [ -n "${BENCH_DEADLINE_FILE:-}" ] && [ -r "$BENCH_DEADLINE_FILE" ]; then
  live=$(tr -d '[:space:]' < "$BENCH_DEADLINE_FILE" 2>/dev/null || true)
  case "$live" in
    ''|*[!0-9]*) ;;
    *) deadline=$live ;;
  esac
fi
if [ -z "$deadline" ]; then
  echo "no benchmark deadline set (BENCH_DEADLINE_FILE and BENCH_DEADLINE_EPOCH are empty)"
  exit 1
fi
now=$(date +%s)
left=$((deadline - now))
if [ "$left" -le 0 ]; then
  echo "deadline reached: the run is being terminated"
  exit 0
fi
printf '%dh %02dm %02ds remaining (deadline in %d s)\n' \
  $((left / 3600)) $((left % 3600 / 60)) $((left % 60)) "$left"
