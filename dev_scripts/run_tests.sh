#!/bin/bash
# Run the test suite.  Works from any directory.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Running pytest on all tests."
# Use `python -m pytest`, not bare `pytest`, so the pytest from the
# active environment is used rather than one earlier on PATH.
python -m pytest -ra "$REPO_ROOT/tests"
