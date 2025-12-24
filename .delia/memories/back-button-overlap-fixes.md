# FloatingBackButton Overlap Fixes

## Root Cause
FloatingBackButton uses absolute positioning at `left: 12px` with 44px width.
On mobile screens, content titles at the top of screens can overlap with this button.

## Pattern to Apply
Add `paddingLeft={56}` (44px button + 12px gap) to header text elements that appear
at the top-left of screens on mobile.

## Screens Fixed

### 1. FavoritesScreen.tsx (lines 269-279)
- Added paddingLeft={56} to XStack containing title
- Added paddingLeft={56} to subtitle TamaguiText

### 2. WelcomeSection.tsx (DashboardScreen) (line 56)
- Added paddingLeft={56} to XStack containing greeting and user name

### 3. MyBookingsScreenPresenter.tsx (lines 98-114)
- Added paddingLeft={56} to XStack containing "Bookings" title
- Added paddingLeft={56} to subtitle text

### 4. GroupBookingSetupPresenter.tsx (lines 117-125)
- Added paddingLeft={56} to YStack containing "Invite Your Group" heading
- Used conditional: `paddingLeft={isLargeScreen ? 0 : 56}` (web wide screens don't need it)

### 5. MyWaiversScreen.tsx (lines 577-582)
- Added paddingLeft={56} to "My Waivers" heading
- Added paddingLeft={56} to subtitle

## Screens That Don't Need Fix
- ProfileScreen: Content is centered
- SearchScreen: No back button (root tab)
- ServiceBookingPresenter: Title is centered (`textAlign="center"`)
- GymListingPresenter: Uses hero image section, not text title at top
- BookingCheckoutPresenter: Uses card-based layout, no overlapping header

## Best Practice Going Forward
When creating new screens with FloatingBackButton:
1. If title appears at top-left on mobile, add `paddingLeft={56}`
2. On large web screens, this padding may not be needed (back button floats outside content)
3. Use conditional: `paddingLeft={Platform.OS === 'web' && width >= 1024 ? 0 : 56}`
