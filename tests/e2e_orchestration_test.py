
import asyncio
import sys
import os
from unittest.mock import MagicMock, patch, AsyncMock

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

from delia.orchestration.service import get_orchestration_service
from delia.orchestration.result import OrchestrationMode, ModelRole

async def run_test():
    print("Initializing OrchestrationService...")
    service = get_orchestration_service()
    
    # Mock the LLM call to return a simple response
    mock_response = {
        "success": True,
        "response": "Hello! I am Delia. How can I help you today?",
        "tokens": 10,
        "model": "test-model"
    }
    
    print("Simulating 'hello' message...")
    
    # We'll use process_stream since that's what 'delia chat' uses
    try:
        async for event in service.process_stream(
            message="hello",
            session_id="test-session",
        ):
            print(f"Event: {event.event_type} - {event.message}")
            if event.event_type == "error":
                print(f"ERROR DETAIL: {event.details}")
    except Exception as e:
        print(f"\nFATAL EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Mocking necessary parts
    with patch("delia.llm.call_llm", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {
            "success": True,
            "response": "Hello! I am Delia.",
            "tokens": 10,
            "model": "test-model"
        }
        
        # Mock backend scorer to avoid metric files
        with patch("delia.routing.BackendScorer.score", return_value=1.0):
            # Mock backend manager to return a fake backend
            mock_backend = MagicMock()
            mock_backend.id = "test-backend"
            mock_backend.name = "Test Backend"
            mock_backend.models = {"quick": "test-model", "coder": "test-model"}
            
            with patch("delia.backend_manager.BackendManager.get_active_backend", return_value=mock_backend):
                with patch("delia.backend_manager.BackendManager.get_enabled_backends", return_value=[mock_backend]):
                    asyncio.run(run_test())
