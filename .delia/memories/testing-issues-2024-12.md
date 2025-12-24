# Testing Issues - December 2024

## Critical Bug Found

### Calendar Date Rendering Bug (Android)
**Priority: HIGH - Blocks Payment Flow**

**Description:**
The booking modal calendar does not render date numbers on Android. Only the month header (e.g., "December 2025") and day-of-week labels (Sun Mon Tue Wed Thu Fri Sat) are displayed. The actual date grid is missing/invisible.

**Steps to Reproduce:**
1. Open any gym listing (e.g., E2E Test Gym)
2. Tap "Book Now" or "Tap to select date & time"
3. Booking modal opens with calendar
4. Calendar shows month and day headers but NO date numbers

**Expected:** Calendar should display clickable date numbers (1-31)
**Actual:** Only headers visible, no dates rendered

**Impact:** 
- Users cannot select a booking date
- Entire booking/payment flow is blocked
- Payment testing cannot proceed

**Device:** Android Emulator (Pixel-style device)
**App Version:** Dev build on dev-clean branch

---

## Database Setup Completed

### Test Gym Dan - Payment Testing Data
- **Database:** dev-clean (mujetufcdoyzytigjfln)
- **Owner:** d.campos+testgym@gorxfitness.com
- **Gym ID:** d9b11cdc-fa43-4996-8a52-ebc342f2804f
- **Location:** New York, NY (40.7128, -74.0060)

**Plans Added:**
| Plan | Type | Price |
|------|------|-------|
| Drop-In Session | DROP_IN | $25.00 |
| Monthly Unlimited | MONTHLY | $89.00 |
| 10-Visit Pack | DROP_IN | $200.00 |

---

## Other Known Issues

1. **Location Services:** Emulator cannot get location (expected)
2. **Google Maps API:** Not configured, Map View crashes
3. **SWR Errors:** gym-plans fetch errors for E2E Test Gym

---

## Completed Testing

- Bottom navigation bar redesign: Working correctly
- All 5 tabs functional (Search, Favorites, Home, Bookings, Profile)
- Group booking toggle: Working
- Search with "Unlimited" radius: Working
- Past bookings display: Working

---

## Next Steps

1. **Fix calendar rendering bug** - Critical priority
2. Re-test payment flow after calendar fix
3. Configure Google Maps API key for Map View testing
