#!/bin/bash
# Hourly cron entry point: lock, then stems + features for the next undone songs (pipeline.py).
# When every song is done: build the report once, then remove this cron entry.
BASE="$HOME/Music/Beat Thief Analysis"
LOCK="$BASE/.lock"
LOG="$BASE/logs/hourly.log"
mkdir -p "$BASE/logs"
if ! mkdir "$LOCK" 2>/dev/null; then
  if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +90 2>/dev/null)" ]; then
    rmdir "$LOCK"; mkdir "$LOCK" || exit 0          # a lock older than 90 min is a dead run
  else
    echo "$(date '+%F %T') busy, skipped" >> "$LOG"; exit 0
  fi
fi
trap 'rmdir "$LOCK"' EXIT
export PATH="/usr/bin:/bin:/opt/homebrew/bin:/usr/local/bin"
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")" || exit 1
echo "$(date '+%F %T') start" >> "$LOG"
if [ ! -e "$BASE/FINISHED" ]; then
  caffeinate -i -s nice -n 10 /usr/bin/python3 pipeline.py --budget-min 45 >> "$LOG" 2>&1
fi
if [ -e "$BASE/all_done" ] && [ ! -e "$BASE/FINISHED" ]; then
  if /usr/bin/python3 report.py >> "$LOG" 2>&1; then
    cp "$BASE/reports/summary.txt" "$BASE/FINISHED"
    /usr/bin/crontab -l 2>/dev/null | grep -v '# beat-analysis$' | /usr/bin/crontab - \
      && echo "$(date '+%F %T') report built, cron entry removed" >> "$LOG" \
      || echo "$(date '+%F %T') report built, could NOT remove cron entry" >> "$LOG"
  fi
fi
echo "$(date '+%F %T') end" >> "$LOG"
