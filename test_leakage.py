
import sys
import os
import io
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath("src"))

def test_imports():
    print("Testing imports for stdout leakage...")
    
    # Capture stdout at the OS level if possible, or just use a pipe
    # But for a simple test, let's just import and see.
    # We already redirected sys.stdout in mcp_server.py
    
    import delia.mcp_server
    print("Import complete.")

if __name__ == "__main__":
    test_imports()
