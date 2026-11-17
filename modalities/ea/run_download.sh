#!/usr/bin/env bash
# Download of articles and supporting information for the elemental analysis dataset (modalities/ea/download_corpus.py).
#
#   bash modalities/ea/run_download.sh start    checks, test download of one article, then starts in the background
#   bash modalities/ea/run_download.sh status   running or not, progress, files on disk, last errors
#   bash modalities/ea/run_download.sh stop     stops it (a new start continues where it stopped)
#   bash modalities/ea/run_download.sh log      follows the log
#
# Settings (environment variables): OUT_DIR, WORKERS, MAX_ARTICLES, PYTHON
set -u
cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-data/ea/corpus2}"
WORKERS="${WORKERS:-64}"
MAX_ARTICLES="${MAX_ARTICLES:-80000}"
PYTHON="${PYTHON:-python3}"
LOG="$OUT_DIR/download.log"
PATTERN="modalities/ea/download_corpus.py --output-dir $OUT_DIR"

ok()   { echo "  [OK]    $*"; }
fail() { echo "  [ERROR] $*"; exit 1; }

running_pid() { pgrep -f "$PATTERN" | head -1; }

count() { find "$1" -name "$2" 2>/dev/null | wc -l | tr -d ' '; }

start() {
    echo "Checks:"
    if [ -n "$(running_pid)" ]; then
        fail "a download into $OUT_DIR is already running (pid $(running_pid)); see: bash modalities/ea/run_download.sh status"
    fi
    "$PYTHON" -c 'import sys; assert sys.version_info >= (3, 8)' 2>/dev/null || fail "$PYTHON not found or older than 3.8 (set PYTHON=...)"
    ok "python: $("$PYTHON" --version 2>&1)"
    command -v pdftotext >/dev/null || fail "pdftotext not found: brew install poppler (macOS) / apt install poppler-utils (Linux)"
    ok "pdftotext: $(command -v pdftotext)"
    for url in "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=test&format=json&pageSize=1" \
               "https://pmc-oa-opendata.s3.amazonaws.com/?list-type=2&max-keys=1" \
               "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi"; do
        code=$(curl -s -o /dev/null -m 30 -w "%{http_code}" "$url")
        [ "$code" = "200" ] || fail "no access to ${url%%\?*} (HTTP $code)"
        host="${url#https://}"; ok "network: ${host%%/*}"
    done
    mkdir -p "$OUT_DIR/raw" "$OUT_DIR/si"
    TEST_DIR=$(mktemp -d)
    "$PYTHON" - "$TEST_DIR" <<'EOF' || fail "test download of one article failed (see the error above)"
import os, sys
sys.path.insert(0, "modalities/ea")
import download_corpus as d
out = sys.argv[1]
for sub in ("raw", "si"):
    os.makedirs(os.path.join(out, sub))
d.fetch({"pmcid": "PMC13284749"}, out)
assert os.path.getsize(os.path.join(out, "raw", "PMC13284749.xml.gz")) > 1000, "empty XML"
assert os.path.getsize(os.path.join(out, "si", "PMC13284749.txt.gz")) > 1000, "no text from the supplementary PDF"
EOF
    rm -rf "$TEST_DIR"
    ok "test article downloaded (XML + supporting information text)"

    SKIP=""; [ -f data/ea/downloaded_pmcids.json ] && SKIP="--skip-ids data/ea/downloaded_pmcids.json"
    SEED=""
    for f in data/ea/found_c/pmcids.json data/ea/si_corpus/si/meta.jsonl; do [ -f "$f" ] && SEED="$SEED $f"; done
    [ -n "$SEED" ] && SEED="--seed$SEED"
    KEEP_AWAKE=""; command -v caffeinate >/dev/null && KEEP_AWAKE="caffeinate -i"
    echo "=== start $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG"
    nohup $KEEP_AWAKE "$PYTHON" -u $PATTERN $SKIP $SEED --max-articles "$MAX_ARTICLES" --num-workers "$WORKERS" >> "$LOG" 2>&1 &
    sleep 20
    if [ -z "$(running_pid)" ]; then
        echo "Download FAILED to start. Last lines of $LOG:"
        tail -15 "$LOG"
        exit 1
    fi
    echo
    echo "Download STARTED (pid $(running_pid)), log: $LOG"
    tail -3 "$LOG"
    echo
    echo "Status:  bash modalities/ea/run_download.sh status"
    echo "Log:     bash modalities/ea/run_download.sh log"
}

status() {
    pid=$(running_pid)
    if [ -n "$pid" ]; then
        echo "State:      RUNNING (pid $pid)"
    elif grep -q "^done" "$LOG" 2>/dev/null; then
        echo "State:      FINISHED"
    else
        echo "State:      NOT RUNNING (start or continue: bash modalities/ea/run_download.sh start)"
    fi
    echo "Full texts: $(count "$OUT_DIR/raw" '*.xml.gz')   supporting information: $(count "$OUT_DIR/si" 'PMC*.txt.gz')   size: $(du -sh "$OUT_DIR" 2>/dev/null | cut -f1)"
    if [ -f "$LOG" ]; then
        echo "Query:      $(grep '^query:' "$LOG" | tail -1)"
        echo "Progress:   $(grep 'download:\|search:' "$LOG" | tail -1 | sed 's/^ *//')"
        errors=$(grep -ci "traceback\|error" "$LOG")
        [ "$errors" -gt 0 ] && { echo "Errors in log: $errors, last:"; grep -i "error" "$LOG" | tail -3; }
    fi
    return 0
}

stop() {
    pid=$(running_pid)
    [ -z "$pid" ] && { echo "Not running"; return 0; }
    pkill -f "$PATTERN"; sleep 2
    [ -z "$(running_pid)" ] && echo "Stopped (bash modalities/ea/run_download.sh start continues from here)" || echo "Still running: kill $(running_pid)"
}

case "${1:-}" in
    start) start ;;
    status) status ;;
    stop) stop ;;
    log) tail -f "$LOG" ;;
    *) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
