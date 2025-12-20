# Setup Simulation Findings

## Test Environments

| Environment | Python | Result |
|-------------|--------|--------|
| Ubuntu 24.04 (clean) | None (uv downloads 3.14) | PASS |
| Ubuntu 24.04 + Python 3.12 | 3.12.3 | PASS |
| Fedora 40 | 3.12.10 | PASS |
| Alpine 3.20 (musl) | None (uv downloads 3.14) | PASS |
| Debian Bullseye (old Python) | 3.9.2 (uv downloads 3.14) | PASS |

## Key Findings

### Positive Discoveries

1. **uv handles Python installation automatically** - Even on systems without Python or with old Python, uv downloads a compatible version. This is a huge UX win.

2. **Works across glibc and musl** - Alpine (musl libc) installs correctly with uv downloading musl-compatible packages.

3. **`delia doctor` provides good diagnostics** - Clear output showing configuration status and backend connectivity.

4. **Installation is fast** - Dependencies install in ~1-2 seconds after download.

### UX Improvement Opportunities

1. **Add `--yes` or `--non-interactive` flag to `delia init`**
   - Current: Interactive wizard that can't be automated
   - Suggestion: Allow `delia init --yes` for CI/scripted setups
   - Priority: Medium

2. **Add `delia version` command**
   - Current: No way to check installed version
   - Suggestion: Add `delia version` or `delia --version`
   - Priority: Low

3. **Consider warning about .venv directory in README**
   - The existing .venv from development causes a warning on first install
   - Not a blocker, but could confuse new users

### Test Failures in Containers

10 tests consistently fail in Docker containers (but pass locally):
- `test_cli.py::TestMainFunction::test_typer_cli_configuration`
- `test_cli.py::TestPackageEntryPoints::test_pyproject_defines_*`
- Some path-related tests

These appear to be test environment issues, not Delia bugs:
- Different Python version (3.14 vs 3.13)
- Path differences in containers

### Common User Pain Points (Anticipated)

1. **"Ollama not reachable"** - Users may forget to start Ollama before Delia
   - Mitigation: `delia doctor` already detects this
   - Suggestion: Add hint to `delia init` output

2. **No MCP clients detected** - Expected in fresh installs
   - Current UX is good - just informational, not an error

## Recommendations

### Short Term
- [ ] Add `--yes`/`--non-interactive` to `delia init`
- [ ] Add `delia version` command

### Medium Term
- [ ] Investigate container test failures (may need Python version pinning)
- [ ] Add integration test with Ollama container

### Long Term
- [ ] Add setup tutorial/wizard improvements
- [ ] Consider `delia doctor --fix` for auto-remediation

## MCP Detection Validation

A common issue is that Delia is not detected as an MCP server by supported applications.
This is typically caused by:

1. **Configuration path issues** - Config written to wrong location
2. **JSON format errors** - Invalid JSON in config file
3. **Command path issues** - `uv` not in PATH when client starts
4. **Protocol errors** - Server doesn't respond to MCP initialize

### Validation Script

```bash
# Run MCP detection validation
./tests/setup-simulation/scripts/validate-mcp-detection.sh

# With verbose output
./tests/setup-simulation/scripts/validate-mcp-detection.sh --verbose
```

### What It Checks

1. **Prerequisites**: uv installed, delia command works
2. **Server Startup**: MCP server can start in STDIO mode
3. **Client Configs**: Delia configured in each client's config file
4. **Protocol Test**: Server responds to MCP initialize message
5. **Tools Export**: All core MCP tools are registered

### Automated Tests

```bash
# Run MCP detection tests
uv run pytest tests/test_mcp_detection.py -v

# Run full MCP protocol tests
uv run pytest tests/test_mcp_protocol.py -v
```

## Running the Tests

```bash
# Quick test (Ubuntu only)
./tests/setup-simulation/run-all-tests.sh --quick

# Full test (all environments)
./tests/setup-simulation/run-all-tests.sh

# Specific environment
./tests/setup-simulation/run-all-tests.sh --env fedora-clean

# MCP detection validation
./tests/setup-simulation/scripts/validate-mcp-detection.sh
```

Requires: Docker or Podman (for setup tests)
