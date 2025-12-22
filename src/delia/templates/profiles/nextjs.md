# Next.js Profile

Load this profile for: Next.js applications, App Router, Server Components, SSR/SSG.

## App Router Structure

```
app/
├── layout.tsx           # Root layout (required)
├── page.tsx             # Home page
├── loading.tsx          # Loading UI
├── error.tsx            # Error boundary
├── not-found.tsx        # 404 page
├── (auth)/              # Route group (no URL impact)
│   ├── login/page.tsx
│   └── register/page.tsx
├── dashboard/
│   ├── layout.tsx       # Nested layout
│   ├── page.tsx
│   └── [id]/page.tsx    # Dynamic route
└── api/
    └── users/route.ts   # API route
```

## Server vs Client Components

```tsx
// Default: Server Component (no directive needed)
async function UserList() {
  const users = await db.getUsers();  // Direct DB access
  return <ul>{users.map(u => <li key={u.id}>{u.name}</li>)}</ul>;
}

// Client Component: Only when needed
'use client';

import { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);
  return <button onClick={() => setCount(c => c + 1)}>{count}</button>;
}
```

## When to Use 'use client'

```
USE 'use client' FOR:
- useState, useEffect, useContext
- Event handlers (onClick, onChange)
- Browser APIs (localStorage, window)
- Third-party client libraries

KEEP AS SERVER COMPONENT:
- Data fetching
- Database/backend access
- Sensitive operations (API keys)
- Large dependencies
```

## Data Fetching

```tsx
// Server Component: Direct async/await
async function Page({ params }: { params: { id: string } }) {
  const data = await fetch(`https://api.example.com/items/${params.id}`, {
    cache: 'force-cache',      // Static (default)
    // cache: 'no-store',      // Dynamic
    // next: { revalidate: 60 } // ISR
  });
  return <div>{data.title}</div>;
}

// Client-side: Use SWR or React Query
'use client';
import useSWR from 'swr';

function UserProfile({ id }: { id: string }) {
  const { data, error, isLoading } = useSWR(`/api/users/${id}`);
  if (isLoading) return <Skeleton />;
  if (error) return <ErrorMessage />;
  return <Profile user={data} />;
}
```

## Server Actions

```tsx
// app/actions.ts
'use server';

import { revalidatePath } from 'next/cache';

export async function createPost(formData: FormData) {
  const title = formData.get('title') as string;

  await db.posts.create({ data: { title } });

  revalidatePath('/posts');
}

// app/posts/new/page.tsx
import { createPost } from '@/app/actions';

export default function NewPost() {
  return (
    <form action={createPost}>
      <input name="title" required />
      <button type="submit">Create</button>
    </form>
  );
}
```

## Metadata & SEO

```tsx
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Page Title',
  description: 'Page description',
  openGraph: {
    title: 'OG Title',
    images: ['/og-image.jpg'],
  },
};

// Dynamic metadata
export async function generateMetadata({ params }): Promise<Metadata> {
  const post = await getPost(params.id);
  return { title: post.title };
}
```

## Performance

```
ALWAYS:
- Use next/image for images (automatic optimization)
- Use next/font for fonts (no layout shift)
- Implement loading.tsx for streaming
- Use Suspense for component-level loading

AVOID:
- Large client bundles
- Unnecessary 'use client' directives
- Blocking data fetches in layouts
```

