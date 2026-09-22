#!/bin/bash
# Install xcal from scratch into a fresh conda environment, then build
# the documentation.  Works from any directory: it locates its own
# folder and the repository root rather than relying on the caller's
# working directory.

set -eo pipefail

NAME="xcal"
PYTHON_VERSION="3.11"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load conda's shell commands (conda activate / conda deactivate).  A
# script runs in a new shell that `conda init` never set up, so these
# functions are otherwise undefined.
source "$(conda info --base)/etc/profile.d/conda.sh"

# Remove any previous builds.
/bin/rm -rf "$REPO_ROOT/docs/build" "$REPO_ROOT/dist" \
            "$REPO_ROOT/$NAME.egg-info" "$REPO_ROOT/build"

# Leave any active environment so the xcal env can be removed and
# recreated (conda refuses to remove the environment it is in).
conda activate base

# Remove the xcal environment if it exists, then delete any leftover
# environment directory a previous failed run may have left behind.
conda env remove -y -n "$NAME" 2>/dev/null || true
rm -rf "$(conda info --base)/envs/$NAME"

# Create and activate a fresh environment.
conda create -y -n "$NAME" python="$PYTHON_VERSION"
conda activate "$NAME"

# Editable install with all developer extras.  Editable (-e) keeps the
# environment pointed at this checkout's code, not a frozen copy.
# Extras: test (pytest), docs (sphinx), spekpy (reflection-tube tests).
pip install -e "$REPO_ROOT[test,docs,spekpy]"

# Build the documentation.
source "$SCRIPT_DIR/build_docs.sh"

red=$(tput setaf 1)
reset=$(tput sgr0)
echo " "
echo "Use"
echo "${red}   conda activate xcal   ${reset}"
echo "to activate the conda environment."
echo " "
