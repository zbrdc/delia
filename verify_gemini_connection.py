import asyncio
import sys
import os
from backend_manager import backend_manager

async def test_connection():
    print("Checking backend health...")
    health = await backend_manager.check_all_health()
    
    print("\nBackend Health Status:")
    for backend_id, is_healthy in health.items():
        status = "✅ ONLINE" if is_healthy else "❌ OFFLINE"
        print(f"- {backend_id}: {status}")

    gemini_backend = backend_manager.get_backend("gemini-cloud")
    if not gemini_backend:
        print("\n❌ Gemini backend 'gemini-cloud' not found in settings.json!")
        return

    print(f"\nTesting Gemini Backend: {gemini_backend.name} ({gemini_backend.url})")
    
    if not health.get(gemini_backend.id):
        print("❌ Gemini backend is offline. Cannot test inference.")
        # Check if API key is set
        if not (gemini_backend.api_key or os.environ.get("GEMINI_API_KEY")):
            print("Hint: GEMINI_API_KEY environment variable is not set or configured in settings.")
        return

    print("\nTesting Gemini inference (simple 'hello'...).\n")
    try:
        import google.generativeai as genai
        
        api_key = gemini_backend.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY is not set. Please set it as an environment variable.")
            return
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        print("Sending request to Gemini...")
        response = await asyncio.to_thread(model.generate_content, "Say hello!")
        
        print("✅ Inference Successful!")
        print(f"Response: {response.text[:200]}...")

    except ImportError:
        print("❌ 'google-generativeai' not installed. Run 'uv add google-generativeai'")
    except Exception as e:
        print(f"❌ Exception during Gemini inference: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
