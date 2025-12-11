import asyncio
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from mcp_server import call_llamacpp, backend_manager

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

    # Send request
    print("Sending 'Hello'...")
    result = await call_llamacpp(
        model=local_backend.models["quick"], # Use the 'quick' model
        prompt="Hello! Are you running on the local GPU?",
        task_type="quick",
        backend_obj=local_backend
    )
    
    if result.get("success"):
        print("\n✅ Response from Local GPU:")
        print(f"{result['response']}")
    else:
        print("\n❌ Failed!")
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    asyncio.run(main())
