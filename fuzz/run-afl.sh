#!/bin/bash
# Thin wrapper - delegates to unified testing-suite runner
# For full options: ~/git/testing-suite/scripts/run-afl.sh --help

TESTING_SUITE="${HOME}/git/testing-suite"

if [[ ! -f "${TESTING_SUITE}/scripts/run-afl.sh" ]]; then
    echo "Error: Unified testing-suite not found at ${TESTING_SUITE}"
    echo "Please clone the testing-suite repository"
    exit 1
fi

exec "${TESTING_SUITE}/scripts/run-afl.sh" --project delia "$@"
