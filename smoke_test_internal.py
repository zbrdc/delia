
import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from delia.container import get_container
from delia.tools.admin import health_impl

async def main():
    print("Initializing container...")
    container = get_container()
    container.initialize()
    
    print("Calling health_impl()...")
    try:
        health = await health_impl()
        print("\nHealth Output:")
        print(health)
    except Exception as e:
        print(f"\nError calling health_impl: {e}")

if __name__ == "__main__":
    asyncio.run(main())
