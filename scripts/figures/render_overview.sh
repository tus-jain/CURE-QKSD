#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
input="$root/figures/fig01_overview.svg"
output="$root/figures/fig01_overview.pdf"
temporary="$(mktemp -d)"
trap 'rm -rf "$temporary"' EXIT

if [[ -n "${CHROME_BIN:-}" ]]; then
  browser="$CHROME_BIN"
elif command -v google-chrome >/dev/null 2>&1; then
  browser="$(command -v google-chrome)"
elif command -v chromium >/dev/null 2>&1; then
  browser="$(command -v chromium)"
elif [[ -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ]]; then
  browser="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
else
  echo "Set CHROME_BIN to a Chrome or Chromium executable." >&2
  exit 1
fi

sed "s|SOURCE_SVG|file://$input|" >"$temporary/overview.html" <<'HTML'
<!doctype html><html><head><meta charset="utf-8"><style>
@page { size: 1500px 1040px; margin: 0; }
html, body { margin: 0; padding: 0; background: #fff; }
img { display: block; width: 1500px; height: 1040px; }
</style></head><body><img src="SOURCE_SVG"></body></html>
HTML

"$browser" --headless --disable-gpu --allow-file-access-from-files --no-pdf-header-footer \
  --print-to-pdf="$output" "file://$temporary/overview.html" >/dev/null 2>&1
