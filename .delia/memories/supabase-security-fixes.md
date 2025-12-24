# Supabase Security Fixes - December 2024

## Completed Fixes

### 1. Function Search Path Security (FIXED)
**Migration:** `fix_function_search_path_security`
**Status:** Applied
**Issue:** 58 functions had mutable search_path, allowing potential SQL injection
**Fix:** Added `SET search_path = ''` to all affected functions in `app` and `public` schemas

### 2. Extensions Moved to Dedicated Schema (FIXED)
**Migration:** `move_extensions_to_extensions_schema`
**Status:** Applied
**Issue:** pg_trgm, btree_gist, vector extensions were in public schema
**Fix:** 
```sql
CREATE SCHEMA IF NOT EXISTS extensions;
ALTER EXTENSION pg_trgm SET SCHEMA extensions;
ALTER EXTENSION btree_gist SET SCHEMA extensions;
ALTER EXTENSION vector SET SCHEMA extensions;
```

## Pending Fixes

### 3. Leaked Password Protection (REQUIRES DASHBOARD)
**Status:** Cannot be fixed via MCP/SQL
**Issue:** Supabase Auth leaked password protection is DISABLED
**Impact:** Users can use passwords known to be compromised (from HaveIBeenPwned)

**How to Fix (Manual Steps):**
1. Go to Supabase Dashboard: https://supabase.com/dashboard/project/mujetufcdoyzytigjfln
2. Navigate to: Authentication → Providers → Email
3. Enable "Leaked Password Protection"
4. Optionally set minimum password strength

**Reference:** https://supabase.com/docs/guides/auth/password-security#password-strength-and-leaked-password-protection

## Verification
After fixes, security advisors show only 1 remaining issue (leaked password protection).
Run `mcp__supabase__get_advisors(project_id, "security")` to verify.
