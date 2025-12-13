#!/bin/bash
# MCP Detection Validation Script
# This script validates that Delia will be properly detected by MCP clients
#
# Usage: ./validate-mcp-detection.sh [--verbose] [--log FILE]

# Don't exit on error - we want to log all issues
# set -e

# Ensure common uv install locations are in PATH
for UV_DIR in "$HOME/.local/bin" "$HOME/.cargo/bin" "/usr/local/bin"; do
    if [[ -x "$UV_DIR/uv" ]] && [[ ":$PATH:" != *":$UV_DIR:"* ]]; then
        export PATH="$UV_DIR:$PATH"
    fi
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

VERBOSE=false
LOG_FILE=""
ERRORS=()
WARNINGS=()
SUCCESSES=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --log|-l)
            LOG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--verbose] [--log FILE]"
            exit 1
            ;;
    esac
done

# Set up logging
if [[ -n "$LOG_FILE" ]]; then
    # Redirect all output to both terminal and log file
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "Logging to: $LOG_FILE"
fi

log_step() {
    echo -e "${BLUE}[CHECK]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
    SUCCESSES+=("$1")
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    WARNINGS+=("$1")
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ERRORS+=("$1")
}

log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_debug() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "       ${NC}$1"
    fi
}

# Always show command output on failure
log_output() {
    local label="$1"
    local output="$2"
    if [[ -n "$output" ]]; then
        echo -e "       ${CYAN}$label:${NC}"
        echo "$output" | sed 's/^/         /'
    fi
}

# ============================================
# Header with timestamp
# ============================================
echo ""
echo "============================================"
echo "MCP Detection Validation"
echo "============================================"
echo ""
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo "PWD:  $(pwd)"
echo ""

# ============================================
# Check 1: Prerequisites
# ============================================
log_step "Checking prerequisites..."

# Check uv is installed
if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version 2>&1 | head -1)
    log_success "uv is installed: $UV_VERSION"
else
    log_error "uv is not installed - MCP clients won't be able to start Delia"
    log_info "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

# Check Python version
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version 2>&1)
    log_success "Python available: $PY_VERSION"
else
    log_warning "python3 not in PATH (uv may still work)"
fi

# Check delia command
DELIA_OUT=$(uv run delia --help 2>&1)
DELIA_EXIT=$?
if [[ $DELIA_EXIT -eq 0 ]]; then
    log_success "'delia' command is accessible via uv"
else
    log_error "'uv run delia' fails - installation may be broken"
    log_output "Output" "$DELIA_OUT"
fi

# Check delia version if available
DELIA_VERSION=$(uv run python -c "from delia import __version__; print(__version__)" 2>/dev/null || echo "N/A")
log_info "Delia version: $DELIA_VERSION"

# ============================================
# Check 2: MCP Server Startup
# ============================================
echo ""
log_step "Checking MCP server can start..."

# Start server and capture ALL output
TEMP_OUT=$(mktemp)
TEMP_ERR=$(mktemp)

# Run with timeout, capture stdout and stderr separately
timeout 3 uv run delia serve > "$TEMP_OUT" 2> "$TEMP_ERR" &
SERVER_PID=$!
sleep 1

SERVER_STDOUT=$(cat "$TEMP_OUT" 2>/dev/null || echo "")
SERVER_STDERR=$(cat "$TEMP_ERR" 2>/dev/null || echo "")

if ps -p $SERVER_PID &> /dev/null 2>&1; then
    log_success "MCP server starts successfully (STDIO mode)"
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
else
    # Process ended - check why
    wait $SERVER_PID 2>/dev/null
    EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 124 ]]; then
        # Timeout - server was waiting for input (good!)
        log_success "MCP server waits for STDIO input (expected behavior)"
    elif [[ $EXIT_CODE -eq 0 ]]; then
        log_success "MCP server exited cleanly"
    else
        log_error "MCP server failed to start (exit code: $EXIT_CODE)"
        if [[ -n "$SERVER_STDOUT" ]]; then
            log_output "stdout" "$SERVER_STDOUT"
        fi
        if [[ -n "$SERVER_STDERR" ]]; then
            log_output "stderr" "$SERVER_STDERR"
        fi
    fi
fi
rm -f "$TEMP_OUT" "$TEMP_ERR"

# ============================================
# Check 3: Configuration Files
# ============================================
echo ""
log_step "Checking MCP client configurations..."

# Define config paths
declare -A CONFIG_PATHS
CONFIG_PATHS["Claude"]="$HOME/.claude/mcp.json"
CONFIG_PATHS["VS Code"]="$HOME/.config/Code/User/mcp.json"
CONFIG_PATHS["VS Code Insiders"]="$HOME/.config/Code - Insiders/User/mcp.json"
CONFIG_PATHS["Cursor"]="$HOME/.cursor/mcp.json"
CONFIG_PATHS["Gemini"]="$HOME/.gemini/settings.json"
CONFIG_PATHS["Windsurf"]="$HOME/.windsurf/mcp.json"
CONFIG_PATHS["Copilot CLI"]="$HOME/.copilot-cli/mcp.json"

CONFIGURED_COUNT=0
NOT_CONFIGURED_COUNT=0
MISSING_COUNT=0

for client in "${!CONFIG_PATHS[@]}"; do
    config_path="${CONFIG_PATHS[$client]}"

    if [[ -f "$config_path" ]]; then
        # Check if delia is configured
        if grep -q '"delia"' "$config_path" 2>/dev/null; then
            # Validate JSON
            JSON_CHECK=$(python3 -c "import json; json.load(open('$config_path'))" 2>&1)
            if [[ $? -eq 0 ]]; then
                log_success "$client: Delia configured in $config_path"
                ((CONFIGURED_COUNT++))

                # Check command path
                if grep -q '"uv"' "$config_path" 2>/dev/null; then
                    log_debug "Using uv command (recommended)"
                else
                    log_warning "$client: Not using 'uv' command - may have PATH issues"
                fi

                # Check directory path exists
                DIR_PATH=$(python3 -c "
import json
with open('$config_path') as f:
    cfg = json.load(f)
    key = 'mcpServers' if 'mcpServers' in cfg else 'servers'
    if key in cfg and 'delia' in cfg[key]:
        args = cfg[key]['delia'].get('args', [])
        if '--directory' in args:
            idx = args.index('--directory')
            if idx + 1 < len(args):
                print(args[idx + 1])
" 2>/dev/null)
                if [[ -n "$DIR_PATH" ]]; then
                    if [[ -d "$DIR_PATH" ]]; then
                        log_debug "Directory exists: $DIR_PATH"
                    else
                        log_error "$client: Configured directory does not exist: $DIR_PATH"
                    fi
                fi
            else
                log_error "$client: Invalid JSON in $config_path"
                log_output "JSON error" "$JSON_CHECK"
            fi
        else
            log_warning "$client: Config exists but Delia not configured"
            log_info "  Path: $config_path"
            ((NOT_CONFIGURED_COUNT++))
        fi
    else
        log_debug "$client: No config file at $config_path"
        ((MISSING_COUNT++))
    fi
done

echo ""
log_info "Config summary: $CONFIGURED_COUNT configured, $NOT_CONFIGURED_COUNT need setup, $MISSING_COUNT clients not installed"

# ============================================
# Check 4: Protocol Test
# ============================================
echo ""
log_step "Testing MCP protocol communication..."

# Create a test that sends an initialize message
PROTO_TEST_RESULT=$(python3 << 'EOF' 2>&1
import subprocess
import json
import sys
import os

# Start the server
proc = subprocess.Popen(
    ["uv", "run", "delia", "serve"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Send initialize request
init_msg = json.dumps({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1.0"}
    }
}) + "\n"

try:
    proc.stdin.write(init_msg)
    proc.stdin.flush()

    # Read response (with timeout)
    import select
    ready, _, _ = select.select([proc.stdout], [], [], 5)
    if ready:
        response = proc.stdout.readline()
        if response:
            try:
                data = json.loads(response)
                if "result" in data:
                    print("PASS: Got valid initialize response")
                    print(f"Protocol version: {data.get('result', {}).get('protocolVersion', 'unknown')}")
                    print(f"Server: {data.get('result', {}).get('serverInfo', {}).get('name', 'unknown')}")
                    sys.exit(0)
                elif "error" in data:
                    print(f"ERROR: Server returned error: {data['error']}")
                    sys.exit(1)
                else:
                    print(f"WARN: Unexpected response format")
                    print(f"Response: {response[:200]}")
                    sys.exit(1)
            except json.JSONDecodeError as e:
                print(f"ERROR: Invalid JSON response: {e}")
                print(f"Raw response: {response[:200]}")
                sys.exit(1)
        else:
            print("WARN: Empty response received")
            sys.exit(1)
    else:
        print("WARN: No response received (timeout after 5s)")
        stderr = proc.stderr.read()
        if stderr:
            print(f"Server stderr: {stderr[:500]}")
        sys.exit(1)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    sys.exit(1)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except:
        proc.kill()
EOF
)
PROTO_EXIT=$?

if [[ $PROTO_EXIT -eq 0 ]]; then
    log_success "MCP protocol test passed"
    log_output "Details" "$PROTO_TEST_RESULT"
else
    log_warning "MCP protocol test did not pass"
    log_output "Details" "$PROTO_TEST_RESULT"
fi

# ============================================
# Check 5: Tools Export
# ============================================
echo ""
log_step "Checking MCP tools are exported..."

TOOLS_RESULT=$(uv run python3 << 'EOF' 2>&1
from delia import mcp_server

core_tools = ["delegate", "think", "batch", "health", "models"]
garden_tools = ["plant", "ponder", "harvest", "prune", "grow", "tend", "ruminate"]
utility_tools = ["switch_backend", "switch_model", "get_model_info_tool", "queue_status"]

missing_core = []
missing_garden = []
missing_utility = []

for tool in core_tools:
    if not hasattr(mcp_server, tool):
        missing_core.append(tool)

for tool in garden_tools:
    if not hasattr(mcp_server, tool):
        missing_garden.append(tool)

for tool in utility_tools:
    if not hasattr(mcp_server, tool):
        missing_utility.append(tool)

print(f"Core tools: {len(core_tools) - len(missing_core)}/{len(core_tools)} exported")
print(f"Garden tools: {len(garden_tools) - len(missing_garden)}/{len(garden_tools)} exported")
print(f"Utility tools: {len(utility_tools) - len(missing_utility)}/{len(utility_tools)} exported")

if missing_core:
    print(f"Missing core: {missing_core}")
    exit(1)
if missing_garden:
    print(f"Missing garden: {missing_garden}")
if missing_utility:
    print(f"Missing utility: {missing_utility}")

exit(0)
EOF
)
TOOLS_EXIT=$?

if [[ $TOOLS_EXIT -eq 0 ]]; then
    log_success "MCP tools exported"
    log_output "Details" "$TOOLS_RESULT"
else
    log_error "Some MCP tools are not exported"
    log_output "Details" "$TOOLS_RESULT"
fi

# ============================================
# Check 6: Settings File
# ============================================
echo ""
log_step "Checking Delia settings..."

SETTINGS_RESULT=$(uv run python3 << 'EOF' 2>&1
import os
from pathlib import Path

# Check for settings file
data_dir = os.environ.get("DELIA_DATA_DIR", str(Path.home() / ".cache" / "delia"))
settings_path = Path(data_dir) / "settings.json"

print(f"Data directory: {data_dir}")
print(f"Settings file: {settings_path}")
print(f"Settings exists: {settings_path.exists()}")

if settings_path.exists():
    import json
    with open(settings_path) as f:
        settings = json.load(f)

    backends = settings.get("backends", [])
    enabled_backends = [b for b in backends if b.get("enabled", False)]

    print(f"Total backends: {len(backends)}")
    print(f"Enabled backends: {len(enabled_backends)}")

    for b in enabled_backends:
        print(f"  - {b.get('name', b.get('id', 'unknown'))}: {b.get('provider', 'unknown')} @ {b.get('url', 'N/A')}")
else:
    print("Settings file not found - run 'delia init' to create")
    exit(1)
EOF
)
SETTINGS_EXIT=$?

if [[ $SETTINGS_EXIT -eq 0 ]]; then
    log_success "Delia settings found"
    log_output "Details" "$SETTINGS_RESULT"
else
    log_warning "Delia settings issue"
    log_output "Details" "$SETTINGS_RESULT"
fi

# ============================================
# Summary
# ============================================
echo ""
echo "============================================"
echo "VALIDATION SUMMARY"
echo "============================================"
echo ""

echo -e "${GREEN}Successes (${#SUCCESSES[@]}):${NC}"
for s in "${SUCCESSES[@]}"; do
    echo -e "  ${GREEN}[OK]${NC} $s"
done

if [[ ${#WARNINGS[@]} -gt 0 ]]; then
    echo ""
    echo -e "${YELLOW}Warnings (${#WARNINGS[@]}):${NC}"
    for w in "${WARNINGS[@]}"; do
        echo -e "  ${YELLOW}[WARN]${NC} $w"
    done
fi

if [[ ${#ERRORS[@]} -gt 0 ]]; then
    echo ""
    echo -e "${RED}Errors (${#ERRORS[@]}):${NC}"
    for e in "${ERRORS[@]}"; do
        echo -e "  ${RED}[FAIL]${NC} $e"
    done
fi

echo ""
echo "============================================"

if [[ ${#ERRORS[@]} -eq 0 ]]; then
    echo -e "${GREEN}RESULT: MCP detection validation PASSED${NC}"
    echo ""
    if [[ ${#WARNINGS[@]} -gt 0 ]]; then
        echo "Note: There are warnings that may need attention."
    fi
    echo ""
    echo "Next steps:"
    echo "  1. Run 'delia install' to configure any unconfigured MCP clients"
    echo "  2. Restart your AI assistant (Claude, VS Code, etc.)"
    echo "  3. Delia tools should appear in the tool list"
    exit 0
else
    echo -e "${RED}RESULT: MCP detection validation FAILED with ${#ERRORS[@]} error(s)${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Ensure uv is installed: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  2. Ensure delia is installed: cd /path/to/delia && uv sync && uv pip install -e ."
    echo "  3. Run 'delia init' to create initial configuration"
    echo "  4. Run 'delia doctor' for more diagnostics"
    echo "  5. Check the output above for specific error details"
    exit 1
fi
