#!/bin/zsh
# Cold/warm timing matrix for one season. Usage: tools/bench_perf.sh <season_dir> [format]
# Cold = ~/.cache/tvshow_noisemaker wiped first (it is only a cache; it refills on the run).
set -euo pipefail
SEASON="$1"; FMT="${2:-mp3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
OUT="$(mktemp -d /tmp/bench_perf.XXXXXX)"
CACHE="$HOME/.cache/tvshow_noisemaker"

if [[ "$(ls -lO "$SEASON" | grep -c dataless)" != "0" ]]; then
  echo "refusing: $SEASON has iCloud-evicted files (download them first)"; exit 1
fi

timings() {  # stage timings from a plan JSON
  "$PY" -c 'import json,sys; t=json.load(open(sys.argv[1]))["stage_timings"]; print("  stage timings:", " ".join(f"{k}={v:.1f}s" for k,v in t.items()))' "$1"
}
run() {  # label, then the CLI args
  local label="$1"; shift
  echo "== $label =="
  { /usr/bin/time -l "$PY" -m tvshow_noisemaker.cli "$@" --benchmark-fingerprint --log-file "$OUT/$label.log" \
      > "$OUT/$label.out" 2> "$OUT/$label.time"; } || true
  grep -E "Fingerprint benchmark|Stage timings|Total runtime|Converted|Failed" "$OUT/$label.out" | sed 's/^/  /' || true
  grep -E "real|user|sys|maximum resident" "$OUT/$label.time" | sed 's/^/  /'
}

rm -rf "$CACHE"
run cold-detect -i "$SEASON" -o "$OUT/cold" --detect-only "$OUT/cold.json"
timings "$OUT/cold.json"
run warm-detect -i "$SEASON" -o "$OUT/warm" --detect-only "$OUT/warm.json"
timings "$OUT/warm.json"
run "warm-encode-$FMT" -i "$SEASON" -o "$OUT/enc-$FMT" --format "$FMT"
echo "outputs in $OUT"
