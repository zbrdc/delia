# Angular Profile

Load this profile for: Angular applications, signals, standalone components, RxJS.

## Standalone Components

```typescript
import { Component, signal, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { UserService } from './user.service';

@Component({
  selector: 'app-user-list',
  standalone: true,
  imports: [CommonModule],
  template: `
    <ul>
      @for (user of users(); track user.id) {
        <li>{{ user.name }}</li>
      }
    </ul>
    <p>Total: {{ count() }}</p>
  `,
})
export class UserListComponent {
  private userService = inject(UserService);

  users = signal<User[]>([]);
  count = computed(() => this.users().length);

  async ngOnInit() {
    const users = await this.userService.getUsers();
    this.users.set(users);
  }
}
```

## Signals (State Management)

```typescript
import { signal, computed, effect } from '@angular/core';

// Writable signal
const count = signal(0);

// Read value
console.log(count());

// Update value
count.set(10);
count.update(n => n + 1);

// Computed (derived state)
const doubled = computed(() => count() * 2);

// Effects (side effects)
effect(() => {
  console.log(`Count is now ${count()}`);
});
```

## File Naming Convention

| Type | Pattern | Example |
|------|---------|---------|
| Component | `*.component.ts` | `user-list.component.ts` |
| Service | `*.service.ts` | `auth.service.ts` |
| Directive | `*.directive.ts` | `highlight.directive.ts` |
| Pipe | `*.pipe.ts` | `format-date.pipe.ts` |
| Guard | `*.guard.ts` | `auth.guard.ts` |
| Test | `*.spec.ts` | `user.service.spec.ts` |

## Services with Dependency Injection

```typescript
import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({ providedIn: 'root' })
export class UserService {
  private http = inject(HttpClient);
  private apiUrl = '/api/users';

  getUsers() {
    return this.http.get<User[]>(this.apiUrl);
  }

  getUser(id: string) {
    return this.http.get<User>(`${this.apiUrl}/${id}`);
  }
}
```

## Control Flow (New Syntax)

```html
<!-- Conditionals -->
@if (isLoading()) {
  <app-spinner />
} @else if (error()) {
  <app-error [message]="error()" />
} @else {
  <app-content [data]="data()" />
}

<!-- Loops with trackBy -->
@for (item of items(); track item.id) {
  <app-item [item]="item" />
} @empty {
  <p>No items found</p>
}

<!-- Deferred loading -->
@defer (on viewport) {
  <app-heavy-component />
} @placeholder {
  <div>Loading...</div>
}
```

## RxJS Patterns

```typescript
import { toSignal } from '@angular/core/rxjs-interop';

@Component({...})
export class SearchComponent {
  private searchService = inject(SearchService);

  searchTerm = signal('');

  // Convert Observable to Signal
  results = toSignal(
    toObservable(this.searchTerm).pipe(
      debounceTime(300),
      distinctUntilChanged(),
      switchMap(term => this.searchService.search(term))
    ),
    { initialValue: [] }
  );
}
```

## Template Best Practices

```
ALWAYS:
- Use async pipe for observables
- Use trackBy with @for
- Use NgOptimizedImage for images
- Keep templates declarative

AVOID:
- Complex logic in templates
- Direct DOM manipulation
- Subscribing in components (use async pipe)
```

