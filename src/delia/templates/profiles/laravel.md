# Laravel Profile

Load this profile for: Laravel applications, PHP, Eloquent ORM, REST APIs.

## Project Structure

```
app/
├── Console/
├── Exceptions/
├── Http/
│   ├── Controllers/
│   ├── Middleware/
│   ├── Requests/          # Form validation
│   └── Resources/         # API transformers
├── Models/
├── Policies/
├── Providers/
├── Repositories/          # Data access layer
└── Services/              # Business logic
config/
database/
├── factories/
├── migrations/
└── seeders/
routes/
├── api.php
└── web.php
tests/
```

## Controllers

```php
<?php

namespace App\Http\Controllers;

use App\Http\Requests\StorePostRequest;
use App\Http\Resources\PostResource;
use App\Models\Post;
use Illuminate\Http\JsonResponse;

final class PostController extends Controller
{
    public function index(): JsonResponse
    {
        $posts = Post::with('author')
            ->published()
            ->paginate(15);

        return PostResource::collection($posts)->response();
    }

    public function store(StorePostRequest $request): JsonResponse
    {
        $post = Post::create([
            ...$request->validated(),
            'user_id' => auth()->id(),
        ]);

        return (new PostResource($post))
            ->response()
            ->setStatusCode(201);
    }

    public function show(Post $post): JsonResponse
    {
        $post->load('author', 'comments.user');

        return (new PostResource($post))->response();
    }

    public function destroy(Post $post): JsonResponse
    {
        $this->authorize('delete', $post);

        $post->delete();

        return response()->json(null, 204);
    }
}
```

## Form Requests

```php
<?php

namespace App\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

final class StorePostRequest extends FormRequest
{
    public function authorize(): bool
    {
        return true;
    }

    public function rules(): array
    {
        return [
            'title' => ['required', 'string', 'max:255'],
            'content' => ['required', 'string'],
            'published' => ['boolean'],
        ];
    }

    public function messages(): array
    {
        return [
            'title.required' => 'A title is required.',
        ];
    }
}
```

## Models & Eloquent

```php
<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;

final class Post extends Model
{
    protected $fillable = [
        'title',
        'content',
        'user_id',
        'published',
    ];

    protected $casts = [
        'published' => 'boolean',
        'published_at' => 'datetime',
    ];

    // Relationships
    public function author(): BelongsTo
    {
        return $this->belongsTo(User::class, 'user_id');
    }

    public function comments(): HasMany
    {
        return $this->hasMany(Comment::class);
    }

    // Scopes
    public function scopePublished($query)
    {
        return $query->where('published', true);
    }

    public function scopeByAuthor($query, int $userId)
    {
        return $query->where('user_id', $userId);
    }
}
```

## API Resources

```php
<?php

namespace App\Http\Resources;

use Illuminate\Http\Resources\Json\JsonResource;

final class PostResource extends JsonResource
{
    public function toArray($request): array
    {
        return [
            'id' => $this->id,
            'title' => $this->title,
            'content' => $this->content,
            'published' => $this->published,
            'author' => new UserResource($this->whenLoaded('author')),
            'comments_count' => $this->whenCounted('comments'),
            'created_at' => $this->created_at->toISOString(),
        ];
    }
}
```

## Query Optimization

```php
// Eager load relationships
$posts = Post::with(['author', 'comments'])->get();

// Select specific columns
$posts = Post::select(['id', 'title', 'user_id'])
    ->with('author:id,name')
    ->get();

// Chunking for large datasets
Post::chunk(100, function ($posts) {
    foreach ($posts as $post) {
        // Process
    }
});

// Transactions
DB::transaction(function () use ($data) {
    $order = Order::create($data);
    $order->items()->createMany($data['items']);
    $this->updateInventory($data['items']);
});
```

## Testing

```php
<?php

namespace Tests\Feature;

use App\Models\Post;
use App\Models\User;
use Tests\TestCase;

final class PostApiTest extends TestCase
{
    public function test_can_list_posts(): void
    {
        Post::factory()->count(3)->create(['published' => true]);

        $response = $this->getJson('/api/posts');

        $response->assertOk()
            ->assertJsonCount(3, 'data');
    }

    public function test_can_create_post(): void
    {
        $user = User::factory()->create();

        $response = $this->actingAs($user)
            ->postJson('/api/posts', [
                'title' => 'Test Post',
                'content' => 'Content here',
            ]);

        $response->assertCreated()
            ->assertJsonPath('data.title', 'Test Post');
    }
}
```

## Best Practices

```
ALWAYS:
- Use Form Requests for validation
- Use API Resources for responses
- Eager load relationships (prevent N+1)
- Use transactions for critical operations
- Add database indexes

AVOID:
- Fat controllers (use services)
- Raw SQL unless necessary
- Skipping authorization checks
- Hardcoding config values
```

