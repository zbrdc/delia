# Svelte Profile

Load this profile for: Svelte 5, SvelteKit, runes, reactive state.

## Svelte 5 Runes

```svelte
<script lang="ts">
  // Reactive state
  let count = $state(0);
  let items = $state<Item[]>([]);

  // Derived values
  let doubled = $derived(count * 2);
  let total = $derived(items.reduce((sum, i) => sum + i.price, 0));

  // Props
  let { title, onClose }: { title: string; onClose: () => void } = $props();

  // Side effects
  $effect(() => {
    console.log(`Count changed to ${count}`);
    // Cleanup function (optional)
    return () => console.log('Cleanup');
  });

  // Bindable props
  let { value = $bindable() }: { value: string } = $props();
</script>

<button onclick={() => count++}>
  Clicked {count} times (doubled: {doubled})
</button>
```

## State Machines (.svelte.ts)

```typescript
// lib/counter.svelte.ts
export function createCounter(initial: number = 0) {
  let count = $state(initial);

  return {
    get count() { return count; },
    increment() { count++; },
    decrement() { count--; },
    reset() { count = initial; },
  };
}

// Usage in component
<script lang="ts">
  import { createCounter } from '$lib/counter.svelte';

  const counter = createCounter(10);
</script>

<button onclick={counter.increment}>{counter.count}</button>
```

## SvelteKit Routing

```
src/routes/
├── +page.svelte           # /
├── +layout.svelte         # Root layout
├── +error.svelte          # Error page
├── about/+page.svelte     # /about
├── blog/
│   ├── +page.svelte       # /blog
│   └── [slug]/
│       ├── +page.svelte   # /blog/:slug
│       └── +page.ts       # Load function
└── api/
    └── users/+server.ts   # API endpoint
```

## Load Functions

```typescript
// +page.ts (universal load)
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ params, fetch }) => {
  const response = await fetch(`/api/posts/${params.slug}`);
  const post = await response.json();
  return { post };
};

// +page.server.ts (server-only load)
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ params, locals }) => {
  const post = await db.getPost(params.slug);
  return { post };
};
```

## Form Actions

```typescript
// +page.server.ts
import type { Actions } from './$types';
import { fail, redirect } from '@sveltejs/kit';

export const actions: Actions = {
  create: async ({ request }) => {
    const data = await request.formData();
    const title = data.get('title');

    if (!title) {
      return fail(400, { error: 'Title required' });
    }

    await db.posts.create({ title });
    throw redirect(303, '/posts');
  },
};
```

```svelte
<!-- +page.svelte -->
<form method="POST" action="?/create">
  <input name="title" required />
  <button>Create</button>
</form>
```

## Styling

```svelte
<style>
  /* Scoped by default */
  .card {
    @apply rounded-lg shadow-md p-4;
  }

  /* Global styles */
  :global(body) {
    margin: 0;
  }
</style>
```

## Performance

```
ALWAYS:
- Use {#key} for forcing re-renders
- Leverage compile-time optimizations
- Use $derived over manual subscriptions

AVOID:
- Unnecessary reactivity
- Large inline event handlers
- Blocking the main thread
```

