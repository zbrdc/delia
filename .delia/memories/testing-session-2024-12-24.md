# Testing Session - December 24, 2024

## Issues Fixed

### 1. WaiverSigningScreen Duplicate Submission Bug
**Files:** `WaiverSigningScreen.native.tsx`, `WaiverSigningScreen.web.tsx`
**Issue:** Creating duplicate waiver submissions causing database constraint errors
**Fix:** Added check for existing waiver status before creating new submissions:
- If completed + active → show "Already Signed" alert
- If pending/sent/opened → reuse existing submission
- Only create new if no existing submission

### 2. Auth Screen Shaking Bug
**File:** `app/components/AuthForm.tsx`
**Issue:** Form shaking/flickering when clicking into/out of email/password fields
**Root Cause:** Early return pattern that completely unmounted the form when `authLoading` was true
**Fix:** 
- Removed early returns for loading states (lines 236-262)
- Added `showLoadingOverlay` variable to track loading state
- Changed to overlay approach - form stays mounted, loading shows on top
- Updated `isAnyLoading` to include auth loading states

**Key Learning:** Never use early returns that unmount forms during loading states. Use overlays instead to prevent mount/unmount cycles that cause layout thrashing.

### 3. Tamagui Spring Animation Config
**File:** `app/theme/tamagui/tamagui.config.ts`
**Issue:** Originally thought shaking was from underdamped spring (damping: 20)
**Note:** Changed damping to 35 for critical damping, but this wasn't the root cause
**Animation kept:** User wanted animations preserved - the real issue was form unmounting

## Test Credentials
From `tests/e2e/fixtures/users.ts`:
- testuser@gorxtest.com / Test123!Pass (regular user)
- gymowner@gorxtest.com / GymOwner123!Pass (gym owner)

## Pending Tests
- Complete booking flow end-to-end
- Gym owner workflows
- User registration/onboarding
- Payment flow completion
