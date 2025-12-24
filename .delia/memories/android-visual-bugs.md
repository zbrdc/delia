# Android Visual Bugs - December 2024

## Critical Layout Issues

### 1. Back Button Overlapping Title Text
**Severity:** High  
**Affected Screens:** Favorites, Home, Profile (when scrolled)
**Issue:** The FloatingBackButton overlays on top of screen titles

**Root Cause Analysis:**
- `FloatingBackButton.tsx:118-122` uses absolute positioning: `position: "absolute"`, `left: 12`
- Screen content titles don't have left padding to account for the 44px button
- Example in `FavoritesScreen.tsx:268-270`:
  ```tsx
  <YStack paddingHorizontal={contentPadding} paddingTop={60}>
    <TamaguiText>Favorites</TamaguiText>  // No paddingLeft!
  ```

**Recommended Fix (Tamagui Best Practice):**
Option A - Add paddingLeft to affected screens:
```tsx
// In screens using FloatingBackButton, add paddingLeft to title row
<XStack paddingLeft={56}> {/* 44px button + 12px gap */}
  <TamaguiText>Favorites</TamaguiText>
</XStack>
```

Option B - Modify FloatingBackButton to take space in layout flow:
```tsx
// Use a wrapper that reserves space without absolute positioning
<YStack height={56}>
  <XStack paddingLeft="$md">
    <BackButton />
  </XStack>
</YStack>
```

**Files to Update:**
- `app/screens/FavoritesScreen.tsx` - Line 269: Add `paddingLeft={56}` to XStack
- `app/screens/DashboardScreen/DashboardPresenter.tsx` - Check header section
- Any screen using `FloatingBackButton` with adjacent title text

### 2. Text Truncation at Screen Bottom
**Severity:** Medium  
**Affected Screens:** Search, Favorites
**Issue:** Content cut off near bottom navigation

**Root Cause:** 
- FlatList/ScrollView not accounting for bottom tab bar height
- Missing `contentContainerStyle.paddingBottom`

**Fix:**
```tsx
<FlatList
  contentContainerStyle={{ paddingBottom: 100 }} // Tab bar + safe area
/>
```

## Minor Issues

### 3. Location Loading State  
**Screen:** Search  
**Issue:** "Getting location..." appears indefinitely on emulator
**Note:** Expected on emulator - consider timeout fallback

### 4. Push Notification Error Toast
**Screen:** All (on login)  
**Issue:** "Failed to register for push notifications" shows on emulator
**Fix:** Suppress in `__DEV__` mode or when Platform.isTV/isEmulator

## Screens Verified OK
- Sign In/Sign Up - Clean layout
- Bookings - No overlap, uses different header pattern
- Profile (top section) - Back button has dedicated row

## Tamagui Layout Best Practices Reference
From nebnet docs:
- Use `XStack` for horizontal layouts with `gap` for spacing
- Use `YStack` for vertical layouts
- Use `useSafeAreaInsets()` hook over SafeAreaView component
- Apply specific insets manually for more control
- KeyboardAvoidingView: Use `behavior="padding"` on iOS, `undefined` on Android
