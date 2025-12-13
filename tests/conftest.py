"""
Pytest configuration for Delia tests.

Automatically isolates test data to prevent pollution of real stats.
"""
import os
import tempfile
import pytest


@pytest.fixture(scope="session", autouse=True)
def isolated_data_dir():
    """Use isolated temp directory for all test data."""
    with tempfile.TemporaryDirectory(prefix="delia-test-") as tmpdir:
        os.environ["DELIA_DATA_DIR"] = tmpdir
        yield tmpdir
