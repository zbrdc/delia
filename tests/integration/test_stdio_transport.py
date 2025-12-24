import json
import os
import subprocess
import sys
import pytest
from pathlib import Path

# Skip if running in CI environments that might not have the full environment set up
@pytest.mark.integration
def test_stdio_jsonrpc_handshake():
    """
    Integration test that spawns the actual delia server process
    and verifies it speaks valid JSON-RPC over stdio.
    """
    
    # We run as a module to handle relative imports correctly
    # Set PYTHONPATH to include src
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").absolute())
    
    # Construct command to run the server as a module
    cmd = [sys.executable, "-m", "delia.mcp_server"]
    
    # Start the process with pipes for stdin/stdout/stderr
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,  # Unbuffered communication
        env=env
    )
    
    try:
        # 1. Send JSON-RPC initialize request
        # This is the standard first message in MCP protocol
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "integration-test",
                    "version": "1.0.0"
                }
            }
        }
        
        # Write request to server's stdin
        json_line = json.dumps(init_request) + "\n"
        process.stdin.write(json_line)
        process.stdin.flush()
        
        # 2. Read response from server's stdout
        # We expect a single line of JSON
        response_line = process.stdout.readline()
        
        # If we got nothing, check stderr for errors
        if not response_line:
            stderr_output = process.stderr.read()
            pytest.fail(f"Server closed stdout unexpectedly. Stderr: {stderr_output}")
            
        # 3. Parse response
        try:
            response = json.loads(response_line)
        except json.JSONDecodeError:
            pytest.fail(f"Server returned invalid JSON: {response_line}")
            
        # 4. Verify response structure
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        result = response["result"]
        assert "protocolVersion" in result
        assert "capabilities" in result
        assert "serverInfo" in result
        assert result["serverInfo"]["name"] == "delia"
        
    finally:
        # Cleanup
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()

if __name__ == "__main__":
    test_stdio_jsonrpc_handshake()
