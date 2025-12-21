# Known Bugs and Regressions

## 1. Routing: `test_select_model_tasks` Failure
- **Status**: Open
- **Description**: The `generate` task type occasionally routes to the `quick` tier instead of the `coder` tier in the test environment.
- **Investigation**: Logic has been simplified and verified via standalone scripts. The failure persists only within the `pytest` environment, suggesting a potential mock leakage or configuration synchronization issue between `delia.config` and the test runtime.
- **Impact**: Low (manual overrides and regex fallbacks still correctly target the coder tier in real-world scenarios).
- **Date Discovered**: 2025-12-21
