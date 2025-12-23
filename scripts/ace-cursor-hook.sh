#!/bin/bash
# ACE Framework Hook for Cursor AI
#
# Cursor 1.7+ supports hooks for agent lifecycle control.
# This hook injects ACE context into agent prompts.
#
# Install by adding to .cursor/settings.json:
# {
#   "cursor.hooks": {
#     "preAgent": ["bash /path/to/ace-cursor-hook.sh"]
#   }
# }
#
# Note: Cursor hooks have different semantics than Claude Code hooks.
# This is a template - adjust based on Cursor's actual hook API.

# Find .delia directory
find_delia_dir() {
    local dir="$PWD"
    while [[ "$dir" != "/" ]]; do
        if [[ -d "$dir/.delia" ]]; then
            echo "$dir/.delia"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

DELIA_DIR=$(find_delia_dir)

if [[ -z "$DELIA_DIR" ]]; then
    # No .delia directory, proceed without modification
    exit 0
fi

# Load project summary
if [[ -f "$DELIA_DIR/project_summary.json" ]]; then
    OVERVIEW=$(jq -r '.overview // empty' "$DELIA_DIR/project_summary.json" 2>/dev/null | head -c 200)
fi

# Load top coding bullets
BULLETS=""
if [[ -f "$DELIA_DIR/playbooks/coding.json" ]]; then
    BULLETS=$(jq -r '.bullets[:5][] | "- " + .content' "$DELIA_DIR/playbooks/coding.json" 2>/dev/null)
fi

# Output ACE context (Cursor may have different output format requirements)
if [[ -n "$BULLETS" ]]; then
    echo "## ACE Framework Context"
    echo ""
    [[ -n "$OVERVIEW" ]] && echo "**Project:** $OVERVIEW"
    echo ""
    echo "**Playbook Guidance:**"
    echo "$BULLETS"
    echo ""
    echo "---"
fi
