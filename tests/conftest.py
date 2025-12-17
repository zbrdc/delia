# Copyright (C) 2024 Delia Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Pytest configuration for Delia tests.

Automatically isolates test data to prevent pollution of real stats.
"""
import os
import tempfile
import pytest
from hypothesis import settings, Verbosity

# Hypothesis profiles for different test scenarios
# Usage: pytest --hypothesis-profile=overnight
settings.register_profile(
    "default",
    max_examples=100,
)
settings.register_profile(
    "quick",
    max_examples=10,
)
settings.register_profile(
    "overnight",
    max_examples=2000,
    verbosity=Verbosity.normal,
)
settings.register_profile(
    "ci",
    max_examples=500,
)

# Load profile from env var or use default
profile = os.environ.get("HYPOTHESIS_PROFILE", "default")
settings.load_profile(profile)


@pytest.fixture(scope="session", autouse=True)
def isolated_data_dir():
    """Use isolated temp directory for all test data AND settings."""
    with tempfile.TemporaryDirectory(prefix="delia-test-") as tmpdir:
        os.environ["DELIA_DATA_DIR"] = tmpdir
        # CRITICAL: Also isolate settings file so tests don't overwrite user config!
        os.environ["DELIA_SETTINGS_FILE"] = os.path.join(tmpdir, "settings.json")
        yield tmpdir


# ========== GPU FUZZ OPTIONS ==========

def pytest_addoption(parser):
    """Add GPU fuzz testing options."""
    parser.addoption(
        "--fuzz-replay",
        type=str,
        choices=["none", "bugs", "coverage", "all"],
        default="none",
        help=(
            "Replay mode for fuzz testing. "
            "'none': Generate new inputs (default). "
            "'bugs': Replay bug-triggering inputs only. "
            "'coverage': Replay coverage-expanding inputs only. "
            "'all': Replay both bugs and coverage inputs."
        ),
    )
    parser.addoption(
        "--fuzz-seed",
        type=int,
        default=None,
        help="Seed for reproducible edge case generation",
    )
    parser.addoption(
        "--fuzz-coverage",
        action="store_true",
        default=False,
        help="Enable branch coverage tracking per input",
    )
    parser.addoption(
        "--fuzz-workers",
        type=int,
        default=4,
        help="Number of parallel workers for streaming mode",
    )


@pytest.fixture
def fuzz_replay(request):
    """Get replay mode (none, bugs, coverage, all)."""
    mode = request.config.getoption("--fuzz-replay", default="none")
    # Support legacy boolean usage (--fuzz-replay without value means 'bugs')
    if mode is True:
        return "bugs"
    return mode


@pytest.fixture
def fuzz_seed(request):
    """Get fuzz seed if provided."""
    return request.config.getoption("--fuzz-seed", default=None)


@pytest.fixture
def fuzz_coverage(request):
    """Check if coverage tracking is enabled."""
    return request.config.getoption("--fuzz-coverage", default=False)


@pytest.fixture
def fuzz_workers(request):
    """Get number of parallel workers."""
    return request.config.getoption("--fuzz-workers", default=4)
