# CSS Profile

Load this profile for: CSS styling, responsive design, animations, accessibility.

## File Organization

```
styles/
├── base/
│   ├── _reset.css
│   ├── _typography.css
│   └── _variables.css
├── components/
│   ├── _buttons.css
│   ├── _cards.css
│   └── _forms.css
├── layouts/
│   ├── _grid.css
│   └── _containers.css
├── utilities/
│   └── _helpers.css
└── main.css
```

## CSS Variables (Custom Properties)

```css
:root {
  /* Colors */
  --color-primary: #3b82f6;
  --color-primary-dark: #2563eb;
  --color-secondary: #64748b;
  --color-success: #22c55e;
  --color-error: #ef4444;

  /* Neutral */
  --color-text: #1f2937;
  --color-text-muted: #6b7280;
  --color-background: #ffffff;
  --color-surface: #f9fafb;
  --color-border: #e5e7eb;

  /* Typography */
  --font-family: system-ui, -apple-system, sans-serif;
  --font-size-xs: 0.75rem;
  --font-size-sm: 0.875rem;
  --font-size-base: 1rem;
  --font-size-lg: 1.125rem;
  --font-size-xl: 1.25rem;
  --font-size-2xl: 1.5rem;

  /* Spacing */
  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-6: 1.5rem;
  --space-8: 2rem;

  /* Borders */
  --radius-sm: 0.25rem;
  --radius-md: 0.375rem;
  --radius-lg: 0.5rem;
  --radius-full: 9999px;

  /* Shadows */
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
  --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);

  /* Transitions */
  --transition-fast: 150ms ease;
  --transition-normal: 250ms ease;
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
  :root {
    --color-text: #f9fafb;
    --color-text-muted: #9ca3af;
    --color-background: #111827;
    --color-surface: #1f2937;
    --color-border: #374151;
  }
}
```

## BEM Naming Convention

```css
/* Block */
.card { }

/* Element */
.card__header { }
.card__body { }
.card__footer { }

/* Modifier */
.card--featured { }
.card--compact { }

/* Example */
.button {
  padding: var(--space-2) var(--space-4);
  border-radius: var(--radius-md);
  font-weight: 500;
  transition: all var(--transition-fast);
}

.button--primary {
  background-color: var(--color-primary);
  color: white;
}

.button--primary:hover {
  background-color: var(--color-primary-dark);
}

.button--outline {
  background-color: transparent;
  border: 1px solid var(--color-border);
}

.button__icon {
  margin-right: var(--space-2);
}
```

## Flexbox Patterns

```css
/* Centered content */
.flex-center {
  display: flex;
  align-items: center;
  justify-content: center;
}

/* Space between */
.flex-between {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

/* Stack (vertical) */
.stack {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}

/* Row with wrap */
.row {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-4);
}

/* Auto-sizing children */
.row > * {
  flex: 1 1 300px;
}
```

## Grid Patterns

```css
/* Responsive grid */
.grid {
  display: grid;
  gap: var(--space-4);
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
}

/* Fixed columns */
.grid-3 {
  display: grid;
  gap: var(--space-4);
  grid-template-columns: repeat(3, 1fr);
}

/* Sidebar layout */
.layout-sidebar {
  display: grid;
  gap: var(--space-6);
  grid-template-columns: 250px 1fr;
}

@media (max-width: 768px) {
  .layout-sidebar {
    grid-template-columns: 1fr;
  }
}
```

## Responsive Design (Mobile-First)

```css
/* Base: Mobile */
.container {
  width: 100%;
  padding: 0 var(--space-4);
}

/* Tablet */
@media (min-width: 768px) {
  .container {
    max-width: 720px;
    margin: 0 auto;
  }
}

/* Desktop */
@media (min-width: 1024px) {
  .container {
    max-width: 960px;
  }
}

/* Large desktop */
@media (min-width: 1280px) {
  .container {
    max-width: 1200px;
  }
}

/* Hide/show utilities */
.hide-mobile {
  display: none;
}

@media (min-width: 768px) {
  .hide-mobile {
    display: block;
  }
  .hide-desktop {
    display: none;
  }
}
```

## Accessibility

```css
/* Focus styles */
:focus-visible {
  outline: 2px solid var(--color-primary);
  outline-offset: 2px;
}

/* Skip link */
.skip-link {
  position: absolute;
  top: -40px;
  left: 0;
  padding: var(--space-2) var(--space-4);
  background: var(--color-primary);
  color: white;
  z-index: 100;
}

.skip-link:focus {
  top: 0;
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}

/* Screen reader only */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
```

## Animations

```css
/* Fade in */
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.animate-fade-in {
  animation: fadeIn var(--transition-normal);
}

/* Slide up */
@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.animate-slide-up {
  animation: slideUp var(--transition-normal);
}

/* Skeleton loading */
@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

.skeleton {
  background: linear-gradient(
    90deg,
    var(--color-surface) 25%,
    var(--color-border) 50%,
    var(--color-surface) 75%
  );
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
}
```

## Best Practices

```
ALWAYS:
- Use CSS variables for theming
- Mobile-first responsive design
- Support prefers-reduced-motion
- Include focus styles for accessibility
- Use rem/em for scalable typography

AVOID:
- !important (document if unavoidable)
- IDs as selectors
- Deep nesting (max 3 levels)
- Inline styles
- Magic numbers (use variables)
```

