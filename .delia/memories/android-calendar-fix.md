# Android Calendar Rendering Fix

## Issue
The react-native-calendars Calendar component inside TamaguiModal with `scrollable={true}` was not rendering date numbers on Android. Only the month header ("December 2025") and day labels (Sun Mon Tue...) were visible.

## Root Cause
React Native ScrollView components require bounded height children. When the Calendar component is inside a ScrollView (from TamaguiModal's scrollable prop), Android doesn't properly calculate the Calendar's height, causing the date grid to collapse to 0 height.

From React Native docs:
> "ScrollViews must have a bounded height in order to work, since they contain unbounded-height children into a bounded container"

## Fix Applied
In `BookingCalendar.tsx`:

1. Added explicit height to the Calendar container YStack:
```tsx
<YStack
  flex={isLargeScreen ? 1 : undefined}
  width={isLargeScreen ? "45%" : "100%"}
  gap="$md"
  // Android: Calendar needs explicit height inside ScrollView/Modal
  {...(Platform.OS === "android" && { minHeight: 380 })}
>
```

2. Added explicit height to the Calendar style:
```tsx
style={{
  // ...existing styles
  // Android: Calendar requires explicit height inside ScrollView/Modal
  ...(Platform.OS === "android" && { height: 350 }),
}}
```

## Files Modified
- `/app/screens/GymListingScreen/components/BookingCalendar.tsx`

## Testing Notes
- iOS and Web don't need this fix (auto-height works)
- Android emulator touch interaction was unreliable during testing
- Fix follows React Native best practices for ScrollView children
