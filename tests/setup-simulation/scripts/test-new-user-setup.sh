#!/bin/bash
# Simulates a new user attempting to set up Delia
# This script captures common pain points and setup issues
#
# Usage: ./test-new-user-setup.sh [--continue-on-error]

# Don't exit on error by default - we want to log all issues
# set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

ERRORS=()
WARNINGS=()
SUCCESS=()

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
    SUCCESS+=("$1")
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    WARNINGS+=("$1")
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    ERRORS+=("$1")
}

log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

# Show command output on failure
log_output() {
    local label="$1"
    local output="$2"
    if [[ -n "$output" ]]; then
        echo -e "       ${CYAN}$label:${NC}"
        echo "$output" | sed 's/^/         /' | head -20
    fi
}

check_command() {
    if command -v "$1" &> /dev/null; then
        log_success "$1 is available"
        return 0
    else
        log_warning "$1 not found"
        return 1
    fi
}

# ============================================
# Header with environment info
# ============================================
echo ""
echo "=========================================="
echo "Delia Setup Simulation"
echo "=========================================="
echo ""
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo "Shell: $SHELL"
echo "PWD:  $(pwd)"
echo ""

# ============================================
# Phase 1: Pre-requisite checks
# ============================================
echo ""
echo "=========================================="
echo "Phase 1: Checking prerequisites"
echo "=========================================="

log_step "Checking for Python..."
if check_command python3; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

    if [[ "$PYTHON_MAJOR" -ge 3 ]] && [[ "$PYTHON_MINOR" -ge 11 ]]; then
        log_success "Python $PYTHON_VERSION meets requirements (>=3.11)"
    else
        log_warning "Python $PYTHON_VERSION is below minimum (3.11 required)"
        log_warning "uv will download a compatible Python version automatically"
    fi
else
    log_warning "Python3 not installed system-wide"
    log_warning "uv will download Python automatically during install"
fi

log_step "Checking for curl (needed for uv install)..."
check_command curl || log_error "curl not available - can't install uv"

log_step "Checking for git..."
check_command git || log_warning "git not available - user can't clone repo easily"

# ============================================
# Phase 2: Installing uv (if needed)
# ============================================
echo ""
echo "=========================================="
echo "Phase 2: Installing uv package manager"
echo "=========================================="

log_step "Checking if uv is installed..."
if ! check_command uv; then
    log_step "Installing uv via official installer..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1; then
        # Source the new path
        export PATH="$HOME/.local/bin:$PATH"
        if check_command uv; then
            log_success "uv installed successfully"
        else
            log_error "uv install script ran but uv not in PATH"
            log_warning "User may need to restart shell or add ~/.local/bin to PATH"
        fi
    else
        log_error "Failed to install uv"
    fi
fi

# ============================================
# Phase 3: Installing Delia
# ============================================
echo ""
echo "=========================================="
echo "Phase 3: Installing Delia"
echo "=========================================="

cd ~/delia || { log_error "Can't find delia directory"; exit 1; }

log_step "Running 'uv sync'..."
UV_SYNC_OUT=$(uv sync 2>&1)
UV_SYNC_EXIT=$?
if [[ $UV_SYNC_EXIT -eq 0 ]]; then
    log_success "Dependencies installed"
else
    log_error "'uv sync' failed (exit code: $UV_SYNC_EXIT)"
    log_output "Output" "$UV_SYNC_OUT"
fi

log_step "Running 'uv pip install -e .'..."
UV_PIP_OUT=$(uv pip install -e . 2>&1)
UV_PIP_EXIT=$?
if [[ $UV_PIP_EXIT -eq 0 ]]; then
    log_success "Delia installed in development mode"
else
    log_error "'uv pip install -e .' failed (exit code: $UV_PIP_EXIT)"
    log_output "Output" "$UV_PIP_OUT"
fi

# ============================================
# Phase 4: Verifying installation
# ============================================
echo ""
echo "=========================================="
echo "Phase 4: Verifying installation"
echo "=========================================="

log_step "Checking if 'delia' command is available..."
DELIA_HELP_OUT=$(uv run delia --help 2>&1)
DELIA_HELP_EXIT=$?
if [[ $DELIA_HELP_EXIT -eq 0 ]]; then
    log_success "'delia' command works"
else
    log_error "'delia' command not found or errors (exit code: $DELIA_HELP_EXIT)"
    log_output "Output" "$DELIA_HELP_OUT"
fi

log_step "Checking delia can show help..."
if [[ $DELIA_HELP_EXIT -eq 0 ]]; then
    log_success "'delia --help' works"
    log_info "Available commands:"
    echo "$DELIA_HELP_OUT" | grep -E "^\s+\w+" | head -10 | sed 's/^/         /'
else
    log_warning "Could not show help"
    log_output "Output" "$DELIA_HELP_OUT"
fi

# ============================================
# Phase 5: First-time setup experience
# ============================================
echo ""
echo "=========================================="
echo "Phase 5: First-time setup (delia init)"
echo "=========================================="

log_step "Testing 'delia init' (may require interaction)..."
# Note: delia init is interactive - we just check it starts
# A real test would need expect/pexpect for full automation
INIT_HELP_OUT=$(timeout 5 uv run delia init --help 2>&1)
INIT_HELP_EXIT=$?
if [[ $INIT_HELP_EXIT -eq 0 ]]; then
    log_success "'delia init' command is available"
    log_warning "Full init test skipped - requires interactive input"
    log_warning "SUGGESTION: Add --non-interactive or --yes flag for CI/automation"
else
    log_error "'delia init' command failed (exit code: $INIT_HELP_EXIT)"
    log_output "Output" "$INIT_HELP_OUT"
fi

# ============================================
# Phase 6: Check for backend connectivity
# ============================================
echo ""
echo "=========================================="
echo "Phase 6: Backend connectivity check"
echo "=========================================="

log_step "Checking if Ollama is reachable (localhost:11434)..."
if curl -s --connect-timeout 5 http://localhost:11434/api/tags &> /dev/null; then
    log_success "Ollama is running and reachable"
else
    log_warning "Ollama not reachable - expected in isolated container"
    log_warning "New users often forget to start Ollama first"
fi

log_step "Running 'delia doctor'..."
DOCTOR_OUT=$(timeout 10 uv run delia doctor 2>&1)
DOCTOR_EXIT=$?
if [[ $DOCTOR_EXIT -eq 0 ]]; then
    log_success "Doctor check completed"
    log_output "Doctor output" "$DOCTOR_OUT"
else
    log_warning "Doctor check failed or timed out (exit code: $DOCTOR_EXIT)"
    log_output "Output" "$DOCTOR_OUT"
fi

# ============================================
# Phase 7: Running tests
# ============================================
echo ""
echo "=========================================="
echo "Phase 7: Running test suite"
echo "=========================================="

log_step "Running pytest..."
PYTEST_OUT=$(timeout 120 uv run pytest tests/ -q --tb=short 2>&1)
PYTEST_EXIT=$?
if [[ $PYTEST_EXIT -eq 0 ]]; then
    log_success "Tests completed"
    # Show summary line
    echo "$PYTEST_OUT" | tail -5 | sed 's/^/         /'
else
    log_warning "Tests failed or timed out (exit code: $PYTEST_EXIT)"
    # Show last 20 lines of output
    log_output "Test output (last 20 lines)" "$(echo "$PYTEST_OUT" | tail -20)"
fi

# ============================================
# Summary Report
# ============================================
echo ""
echo "=========================================="
echo "SETUP SIMULATION REPORT"
echo "=========================================="

echo ""
echo -e "${GREEN}Successes (${#SUCCESS[@]}):${NC}"
for s in "${SUCCESS[@]}"; do
    echo "  - $s"
done

echo ""
echo -e "${YELLOW}Warnings (${#WARNINGS[@]}):${NC}"
for w in "${WARNINGS[@]}"; do
    echo "  - $w"
done

echo ""
echo -e "${RED}Errors (${#ERRORS[@]}):${NC}"
for e in "${ERRORS[@]}"; do
    echo "  - $e"
done

echo ""
if [[ ${#ERRORS[@]} -eq 0 ]]; then
    echo -e "${GREEN}RESULT: Setup simulation PASSED${NC}"
    exit 0
else
    echo -e "${RED}RESULT: Setup simulation FAILED with ${#ERRORS[@]} error(s)${NC}"
    exit 1
fi
