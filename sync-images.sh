#!/usr/bin/env bash
# Local preview only: fetch the figures from gh-pages:static/, their single source of truth.
# In CI the website deploy links images/ to static/ instead.
# Usage: ./sync-images.sh [ref]   (default origin/gh-pages; pass gh-pages for unpushed local figures)
set -euo pipefail
cd "$(dirname "$0")"
ref="${1:-origin/gh-pages}"
[ "$ref" = "origin/gh-pages" ] && git fetch -q origin gh-pages
rm -rf images && mkdir images
git archive "$ref" static | tar -x -C images --strip-components=1
