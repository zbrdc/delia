# Bottom Navigation Redesign Plan

## Current Issues
1. **Flat appearance** - No elevation, blur, or visual separation from content
2. **Small icons** (24px) and tiny text (`fontSize: "$xs"`)
3. **Outdated active state** - Just a small 4px yellow dot
4. **LinearGradient hack** at top doesn't look modern
5. **No glass/blur effect** - Missing iOS-style translucent bar

## Design Improvements Needed

### Visual Style (2025 Modern)
- Add `expo-blur` for glass effect (stable on Android with SDK 55+, project uses SDK 52)
- Larger icons: 28-32px with filled/outline variants for active/inactive
- Larger, readable labels
- Floating pill-style active indicator instead of dot
- Subtle shadow/elevation
- Remove LinearGradient, use clean blur instead

### Implementation References
- `expo-blur` - BlurView component for tab bars
- `expo-glass-effect` - Newer glass morphism API
- `react-native-bottom-tabs` - Native SwiftUI/Material design tabs

### File Location
`/app/components/navigation/tamagui/TamaguiBottomNavigation.tsx`

### Key Changes
```tsx
// 1. Add expo-blur import
import { BlurView } from 'expo-blur'

// 2. Wrap container in BlurView
<BlurView intensity={80} tint="dark" style={styles.container}>
  <NavContent>...</NavContent>
</BlurView>

// 3. Increase icon size to 28px
// 4. Replace ActiveDot with pill-shaped indicator behind active item
// 5. Add subtle elevation/shadow
// 6. Make labels larger and more readable
```

### Accessibility
- Touch targets remain 44x44px minimum
- Clear active/inactive visual distinction
- Haptic feedback already implemented ✓
