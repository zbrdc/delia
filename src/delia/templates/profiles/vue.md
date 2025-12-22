# Vue.js Profile

Load this profile for: Vue 3, Composition API, Nuxt, Pinia state management.

## Composition API Patterns

```vue
<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';

// Props with defaults
const props = withDefaults(defineProps<{
  title: string;
  count?: number;
}>(), {
  count: 0,
});

// Emits
const emit = defineEmits<{
  (e: 'update', value: number): void;
  (e: 'close'): void;
}>();

// Reactive state
const isLoading = ref(false);
const items = ref<Item[]>([]);

// Computed
const total = computed(() => items.value.length);

// Methods
async function fetchItems() {
  isLoading.value = true;
  try {
    items.value = await api.getItems();
  } finally {
    isLoading.value = false;
  }
}

// Lifecycle
onMounted(fetchItems);
</script>

<template>
  <div>
    <h1>{{ title }}</h1>
    <span v-if="isLoading">Loading...</span>
    <ul v-else>
      <li v-for="item in items" :key="item.id">{{ item.name }}</li>
    </ul>
  </div>
</template>
```

## Naming Conventions

| Element | Case | Example |
|---------|------|---------|
| Components | PascalCase | `UserCard.vue` |
| Composables | camelCase + use | `useAuth.ts` |
| Directories | lowercase-dash | `user-profile/` |
| Props/Events | camelCase | `isActive`, `onUpdate` |

## Composables

```typescript
// composables/useUser.ts
import { ref, computed } from 'vue';

export function useUser(userId: string) {
  const user = ref<User | null>(null);
  const isLoading = ref(true);
  const error = ref<Error | null>(null);

  const fullName = computed(() =>
    user.value ? `${user.value.firstName} ${user.value.lastName}` : ''
  );

  async function fetch() {
    isLoading.value = true;
    try {
      user.value = await api.getUser(userId);
    } catch (e) {
      error.value = e as Error;
    } finally {
      isLoading.value = false;
    }
  }

  return { user, isLoading, error, fullName, fetch };
}
```

## Pinia Store

```typescript
// stores/auth.ts
import { defineStore } from 'pinia';

export const useAuthStore = defineStore('auth', () => {
  const user = ref<User | null>(null);
  const isAuthenticated = computed(() => !!user.value);

  async function login(credentials: LoginCredentials) {
    user.value = await api.login(credentials);
  }

  function logout() {
    user.value = null;
  }

  return { user, isAuthenticated, login, logout };
});
```

## Nuxt 3 Specifics

```vue
<script setup lang="ts">
// Auto-imported composables
const { data: posts, pending } = await useFetch('/api/posts');

// SEO
useSeoMeta({
  title: 'Page Title',
  description: 'Page description',
});

// Runtime config
const config = useRuntimeConfig();
</script>
```

## Template Best Practices

```
ALWAYS:
- Use v-for with :key
- Use v-if over v-show for rare toggles
- Keep templates clean, move logic to composables
- Use scoped styles

AVOID:
- v-if and v-for on same element
- Complex expressions in templates
- Mutating props directly
```

