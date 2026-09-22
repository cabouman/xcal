#!/bin/bash
# Purge the documentation and rebuild it.  Works from any directory.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

/bin/rm -rf "$REPO_ROOT/docs/build"

make -C "$REPO_ROOT/docs" clean html

echo ""
echo "*** The html documentation is at $REPO_ROOT/docs/build/html/index.html ***"
echo ""
