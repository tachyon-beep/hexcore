# tests/test_mtg_data_loader.py

"""
DEPRECATED: This file has been moved to tests/data/test_mtg_data_loader.py

This test file has been relocated to better align with the project's
test organization structure, grouping tests by their component responsibility.

Please update your imports and references to use the new location:
    from tests.data.test_mtg_data_loader import *

This file will be removed in a future update.
"""

import sys
import warnings
from pathlib import Path

warnings.warn(
    "tests/test_mtg_data_loader.py is deprecated. Use tests/data/test_mtg_data_loader.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Only forward test execution to maintain backward compatibility
if __name__ != "__main__":
    # Import the pytest fixtures to make them available
    from tests.data.test_mtg_data_loader import data_loader, data_paths

    # Now import the test functions one by one
    from tests.data.test_mtg_data_loader import test_load_cards
    from tests.data.test_mtg_data_loader import test_load_rules
    from tests.data.test_mtg_data_loader import test_create_documents
    from tests.data.test_mtg_data_loader import test_card_lookup
    from tests.data.test_mtg_data_loader import test_rule_lookup
    from tests.data.test_mtg_data_loader import test_card_search
    from tests.data.test_mtg_data_loader import test_rule_search
else:
    # If this file is run directly as a script, run the actual tests from the new location
    import pytest

    sys.exit(pytest.main(["-xvs", "tests/data/test_mtg_data_loader.py"]))
