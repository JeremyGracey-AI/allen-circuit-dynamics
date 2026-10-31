#!/usr/bin/env bash
# Parallel ranged download: pdl.sh <url> <out> [parts=16]
# For links that cap each TCP connection (~200 kB/s per stream) but not the aggregate.
# Each part is resumable; rerun to resume. Verifies total size against Content-Length.
set -euo pipefail
URL="$1"; OUT="$2"; PARTS="${3:-16}"
SIZE=$(curl -sSIL "$URL" | tr -d '\r' | awk 'tolower($1)=="content-length:"{s=$2} END{print s}')
[[ -n "$SIZE" ]] || { echo "could not read Content-Length"; exit 1; }
TMP="${OUT}.parts"; mkdir -p "$TMP"
CH=$(( (SIZE + PARTS - 1) / PARTS ))
echo "pdl: $SIZE bytes in $PARTS parts of ~$CH -> $OUT"
pids=()
for ((i=0; i<PARTS; i++)); do
  s=$(( i * CH )); e=$(( s + CH - 1 )); (( e >= SIZE )) && e=$(( SIZE - 1 ))
  (( s > e )) && continue
  p="$TMP/$(printf '%03d' "$i")"
  want=$(( e - s + 1 ))
  have=0; [[ -f "$p" ]] && have=$(stat -c %s "$p")
  (( have >= want )) && continue
  ( curl -sS -L --retry 30 --retry-delay 3 --retry-all-errors -r "$(( s + have ))-$e" "$URL" >> "$p" ) &
  pids+=($!)
done
fail=0; for pid in "${pids[@]:-}"; do [[ -n "$pid" ]] && { wait "$pid" || fail=1; }; done
(( fail == 0 )) || { echo "pdl: some parts failed; rerun to resume"; exit 1; }
cat "$TMP"/[0-9][0-9][0-9] > "$OUT"
got=$(stat -c %s "$OUT")
if [[ "$got" != "$SIZE" ]]; then echo "pdl: size mismatch ($got != $SIZE); rerun"; exit 1; fi
rm -rf "$TMP"; echo "pdl: done, $got bytes"
