#!/bin/bash
# Run setup simulation tests across multiple environments
# Usage: ./run-all-tests.sh [--quick] [--env ENV_NAME]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Container runtime detection
if command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
elif command -v docker &> /dev/null; then
    CONTAINER_CMD="docker"
else
    echo -e "${RED}Error: Neither podman nor docker found${NC}"
    exit 1
fi

echo -e "${BLUE}Using container runtime: $CONTAINER_CMD${NC}"

# Parse arguments
QUICK_MODE=false
SPECIFIC_ENV=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --env)
            SPECIFIC_ENV="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quick] [--env ENV_NAME]"
            exit 1
            ;;
    esac
done

# Define test environments
if [[ "$QUICK_MODE" == "true" ]]; then
    ENVIRONMENTS=("ubuntu-clean")
elif [[ -n "$SPECIFIC_ENV" ]]; then
    ENVIRONMENTS=("$SPECIFIC_ENV")
else
    ENVIRONMENTS=(
        "ubuntu-clean"
        "ubuntu-python-only"
        "fedora-clean"
        "alpine-minimal"
        "debian-oldpython"
    )
fi

# Results tracking
declare -A RESULTS
TOTAL_PASS=0
TOTAL_FAIL=0

run_test() {
    local env_name=$1
    local dockerfile="$SCRIPT_DIR/dockerfiles/Dockerfile.$env_name"
    local image_name="delia-setup-test-$env_name"

    echo ""
    echo "========================================"
    echo -e "${BLUE}Testing: $env_name${NC}"
    echo "========================================"

    if [[ ! -f "$dockerfile" ]]; then
        echo -e "${RED}Dockerfile not found: $dockerfile${NC}"
        RESULTS[$env_name]="SKIPPED"
        return 1
    fi

    # Build the image
    echo -e "${BLUE}Building image...${NC}"
    if ! $CONTAINER_CMD build -t "$image_name" -f "$dockerfile" "$PROJECT_ROOT" 2>&1 | tail -10; then
        echo -e "${RED}Build failed for $env_name${NC}"
        RESULTS[$env_name]="BUILD_FAILED"
        ((TOTAL_FAIL++))
        return 1
    fi

    # Run the setup test
    echo -e "${BLUE}Running setup simulation...${NC}"

    # Create a temp script that will be run in container
    local test_script="$SCRIPT_DIR/scripts/test-new-user-setup.sh"

    # Run script via bash, reading from stdin to avoid permission issues
    if $CONTAINER_CMD run --rm -i \
        "$image_name" \
        bash < "$test_script" 2>&1; then
        echo -e "${GREEN}$env_name: PASSED${NC}"
        RESULTS[$env_name]="PASSED"
        ((TOTAL_PASS++))
    else
        echo -e "${RED}$env_name: FAILED${NC}"
        RESULTS[$env_name]="FAILED"
        ((TOTAL_FAIL++))
    fi

    # Cleanup image (optional, saves disk space)
    # $CONTAINER_CMD rmi "$image_name" &> /dev/null || true
}

# Header
echo ""
echo "=============================================="
echo "  Delia Setup Simulation Test Suite"
echo "=============================================="
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Environments to test: ${ENVIRONMENTS[*]}"
echo ""

# Run tests for each environment
for env in "${ENVIRONMENTS[@]}"; do
    run_test "$env" || true  # Continue even if one fails
done

# Summary
echo ""
echo "=============================================="
echo "  FINAL SUMMARY"
echo "=============================================="
echo ""

for env in "${!RESULTS[@]}"; do
    case ${RESULTS[$env]} in
        PASSED)
            echo -e "  ${GREEN}✓${NC} $env"
            ;;
        FAILED)
            echo -e "  ${RED}✗${NC} $env"
            ;;
        BUILD_FAILED)
            echo -e "  ${RED}✗${NC} $env (build failed)"
            ;;
        SKIPPED)
            echo -e "  ${YELLOW}○${NC} $env (skipped)"
            ;;
    esac
done

echo ""
echo "Total: $TOTAL_PASS passed, $TOTAL_FAIL failed"
echo ""

if [[ $TOTAL_FAIL -gt 0 ]]; then
    echo -e "${RED}Some tests failed - review the output above for details${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
