# Android Booking Flow Bugs

## Critical Bug: Calendar Dates Not Rendering

**Location:** BookingSheet modal on Android (E2E Test Gym)
**Component:** `react-native-calendars` Calendar inside TamaguiModal

### Symptoms:
- Calendar shows month header "December 2025"
- Calendar shows day labels (Sun Mon Tue Wed Thu Fri Sat)
- **Calendar date grid (actual numbered dates) is NOT visible**
- Scrolling within modal does not reveal the dates

### Affected Files:
- `app/screens/GymListingScreen/components/BookingSheet.tsx`
- `app/screens/GymListingScreen/components/BookingCalendar.tsx`
- `app/components/tamagui/TamaguiModal.tsx`

### Possible Causes:
1. Height constraint issue - modal content may be clipping calendar
2. `react-native-calendars` rendering issue on Android with centered modal
3. Nested ScrollView conflict (modal is scrollable + calendar internal scroll)
4. Missing explicit height on Calendar component

### Investigation Needed:
- Check if calendar renders correctly outside of modal
- Test with fixed height on Calendar component
- Check TamaguiModal height constraints
- Test with different `presentation` prop values

## Other Observations:

### SWR Error on Gym Detail Page
- Saw error toast: `[SWR] ❌ Error: gym-plans.e2e00000-0...`
- Indicates gym plans fetch is failing for test gym
- May be RLS or data issue

### Back Button Overlap (Fixed)
- Multiple screens had back button overlapping title text
- Fixed by adding paddingLeft={56} to header text elements
- See `back-button-overlap-fixes.md` for full list
