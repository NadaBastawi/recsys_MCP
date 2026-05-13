"""pytest configuration for TORCO Recommendation System tests"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Set up test environment before running tests."""
    # Change to project root directory
    os.chdir(project_root)
    yield


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark tests that require external services
        if "api_client" in item.fixturenames:
            item.add_marker(pytest.mark.integration)
