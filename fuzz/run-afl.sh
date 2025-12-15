#!/bin/bash
# AFL++ fuzzing wrapper for Delia
# Handles environment setup and provides sensible defaults

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Environment variables for Python AFL++
export AFL_SKIP_BIN_CHECK=1
export AFL_DUMB_FORKSRV=1

# Default paths
CORPUS_DIR="${SCRIPT_DIR}/corpus"
FINDINGS_DIR="${SCRIPT_DIR}/findings"
HARNESS="${SCRIPT_DIR}/afl_harness.py"

# Parse arguments
PERSISTENT=""
TARGET=""
MULTI_CORE=""
EXTRA_ARGS=""

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

AFL++ fuzzing wrapper for Delia with automatic environment setup.

Options:
  --persistent, -p     Run in persistent mode (10-100x faster)
  --target, -t NAME    Fuzz specific target
  --multi, -m          Multi-core fuzzing (main + secondary)
  --check              Check system config without fuzzing
  --fix                Attempt to fix system config (needs sudo)
  --help, -h           Show this help

Targets (set via AFL_TARGET env var):
  all          Combined fuzzing (default)
  parser       Tool call parsing (parse_tool_calls)
  code_detect  Code detection (detect_code_content)
  routing      Model override parsing
  validation   Input validation functions
  tool_calls   Text-mode tool parsing
  file_path    Path validation (security critical)

Examples:
  ./run-afl.sh --persistent                  # Fast persistent mode
  ./run-afl.sh --target parser --persistent  # Parser only
  ./run-afl.sh --multi                       # Multi-core fuzzing
  ./run-afl.sh --check                       # Check system state
EOF
    exit 0
}

NEEDS_CORE_FIX=0
NEEDS_GOV_FIX=0

check_system() {
    echo -e "${BLUE}[*]${NC} Checking system configuration..."

    # Check core_pattern
    local pattern=$(cat /proc/sys/kernel/core_pattern 2>/dev/null)
    if [[ "$pattern" == "core"* ]]; then
        echo -e "${GREEN}[+]${NC} core_pattern: OK ($pattern)"
    else
        echo -e "${YELLOW}[!]${NC} core_pattern: $pattern"
        echo "    Fix: echo core | sudo tee /proc/sys/kernel/core_pattern"
        export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
        NEEDS_CORE_FIX=1
    fi

    # Check CPU governor
    local gov_path="/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    if [[ -f "$gov_path" ]]; then
        local gov=$(cat "$gov_path")
        if [[ "$gov" == "performance" ]]; then
            echo -e "${GREEN}[+]${NC} CPU governor: performance"
        else
            echo -e "${YELLOW}[!]${NC} CPU governor: $gov (recommend: performance)"
            echo "    Fix: sudo cpupower frequency-set -g performance"
            NEEDS_GOV_FIX=1
        fi
    fi

    # Check python-afl
    if python3 -c "import afl" 2>/dev/null; then
        echo -e "${GREEN}[+]${NC} python-afl: installed"
    else
        echo -e "${RED}[-]${NC} python-afl: NOT installed"
        echo "    Fix: pip install python-afl"
    fi

    # Check py-afl-fuzz
    if command -v py-afl-fuzz &>/dev/null; then
        echo -e "${GREEN}[+]${NC} py-afl-fuzz: $(command -v py-afl-fuzz)"
    else
        echo -e "${RED}[-]${NC} py-afl-fuzz: NOT found"
        echo "    Fix: pip install python-afl"
    fi
}

fix_system() {
    echo -e "${BLUE}[*]${NC} Attempting to fix system configuration..."

    # Fix core_pattern
    local pattern=$(cat /proc/sys/kernel/core_pattern 2>/dev/null)
    if [[ "$pattern" != "core"* ]]; then
        echo -e "${BLUE}[*]${NC} Setting core_pattern..."
        if [[ $EUID -eq 0 ]]; then
            echo core > /proc/sys/kernel/core_pattern
            echo -e "${GREEN}[+]${NC} core_pattern fixed"
        elif sudo sh -c 'echo core > /proc/sys/kernel/core_pattern' 2>/dev/null; then
            echo -e "${GREEN}[+]${NC} core_pattern fixed"
        else
            echo -e "${YELLOW}[!]${NC} Could not fix core_pattern (run with sudo)"
        fi
    fi

    # Fix CPU governor
    local gov_path="/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    if [[ -f "$gov_path" ]]; then
        local gov=$(cat "$gov_path")
        if [[ "$gov" != "performance" ]]; then
            echo -e "${BLUE}[*]${NC} Setting CPU governor to performance..."
            if [[ $EUID -eq 0 ]]; then
                for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
                    echo performance > "$cpu" 2>/dev/null
                done
                echo -e "${GREEN}[+]${NC} CPU governor fixed"
            elif sudo cpupower frequency-set -g performance 2>/dev/null; then
                echo -e "${GREEN}[+]${NC} CPU governor fixed"
            else
                echo -e "${YELLOW}[!]${NC} Could not fix CPU governor"
            fi
        fi
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --persistent|-p)
            PERSISTENT="--persistent"
            shift
            ;;
        --target|-t)
            export AFL_TARGET="$2"
            TARGET="$2"
            shift 2
            ;;
        --multi|-m)
            MULTI_CORE=1
            shift
            ;;
        --check)
            echo ""
            check_system
            echo ""
            exit 0
            ;;
        --fix)
            echo ""
            fix_system
            echo ""
            check_system
            echo ""
            exit 0
            ;;
        --help|-h)
            show_help
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Create findings directory
mkdir -p "$FINDINGS_DIR"

# Run system check (sets AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES if needed)
echo ""
check_system
echo ""

# Show configuration
echo -e "${BLUE}[*]${NC} Starting AFL++ fuzzing..."
echo "    Project: Delia"
[[ -n "$TARGET" ]] && echo "    Target: $TARGET"
[[ -n "$PERSISTENT" ]] && echo "    Mode: Persistent (fast)"
echo "    Corpus: $CORPUS_DIR"
echo "    Findings: $FINDINGS_DIR"
echo ""

cd "$PROJECT_DIR"

if [[ $MULTI_CORE -eq 1 ]]; then
    echo -e "${BLUE}[*]${NC} Starting multi-core fuzzing..."
    echo "    Run additional instances with: AFL_TARGET=$TARGET py-afl-fuzz -S sec1 -i $CORPUS_DIR -o $FINDINGS_DIR -- python3 $HARNESS $PERSISTENT"
    exec py-afl-fuzz -M main -i "$CORPUS_DIR" -o "$FINDINGS_DIR" $EXTRA_ARGS -- python3 "$HARNESS" $PERSISTENT
else
    exec py-afl-fuzz -i "$CORPUS_DIR" -o "$FINDINGS_DIR" $EXTRA_ARGS -- python3 "$HARNESS" $PERSISTENT
fi
