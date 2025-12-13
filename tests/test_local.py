# Copyright (C) 2023 the project owner
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
"""Manual test script for local GPU (llama.cpp) connectivity.

Run directly: python tests/test_local.py
"""
import asyncio
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from delia.backend_manager import backend_manager
from delia.config import config
from delia.providers import LlamaCppProvider


async def main():
    print("Sending message to Local GPU (llama.cpp)...")

    # Ensure backend manager loads settings
    backend_manager._load_settings()

    # Get the local backend config
    local_backend = backend_manager.get_backend("llamacpp-local")

    if not local_backend:
        print("Error: 'llamacpp-local' backend not found in settings.")
        return

    if not local_backend.enabled:
        print("Error: 'llamacpp-local' is disabled.")
        return

    # Check health first
    print(f"Checking health of {local_backend.url}...")
    is_healthy = await local_backend.check_health()
    if not is_healthy:
        print("❌ Local backend is OFFLINE. Is llama-server running?")
        return

    # Create provider instance
    provider = LlamaCppProvider(
        config=config,
        backend_manager=backend_manager,
    )

    # Send request
    print("Sending 'Hello'...")
    response = await provider.call(
        model=local_backend.models["quick"],  # Use the 'quick' model
        prompt="Hello! Are you running on the local GPU?",
        task_type="quick",
        backend_obj=local_backend,
    )

    result = response.to_dict()
    if result.get("success"):
        print("\n✅ Response from Local GPU:")
        print(f"{result['response']}")
    else:
        print("\n❌ Failed!")
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())
