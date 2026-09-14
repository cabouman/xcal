#!/bin/bash
# Run the test suite.

echo "Running pytest on all tests."
# Use `python -m pytest`, not bare `pytest`, so the pytest from the
# active environment is used rather than one earlier on PATH.
python -m pytest -ra ../tests
