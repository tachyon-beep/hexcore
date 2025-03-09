#!/usr/bin/env python
# tests/test_harness.py
"""
DEPRECATED: This file has been migrated to src/tools/mtg_test_runner.py

This test harness was not an actual test file but a utility for running tests.
It has been relocated to the appropriate tools directory with improved documentation.

Please update your scripts to use the new path:
python -m src.tools.mtg_test_runner [arguments]

This file will be removed in a future update.
"""

import sys
import warnings

warnings.warn(
    "tests/test_harness.py is deprecated. Use src/tools/mtg_test_runner.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    sys.path.append(".")
    from src.tools.mtg_test_runner import main

    main()
