# Django Profile

Load this profile for: Django applications, REST APIs, ORM, authentication.

## Project Structure

```
project/
├── config/
│   ├── __init__.py
│   ├── settings/
│   │   ├── base.py
│   │   ├── development.py
│   │   └── production.py
│   ├── urls.py
│   └── wsgi.py
├── apps/
│   └── users/
│       ├── __init__.py
│       ├── admin.py
│       ├── models.py
│       ├── views.py
│       ├── serializers.py
│       ├── urls.py
│       └── tests.py
├── manage.py
└── requirements.txt
```

## Models

```python
from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    email = models.EmailField(unique=True)
    bio = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.email


class Post(models.Model):
    author = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='posts'
    )
    title = models.CharField(max_length=200)
    content = models.TextField()
    published = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['author', 'published']),
        ]
```

## Django REST Framework Views

```python
from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response

class PostViewSet(viewsets.ModelViewSet):
    queryset = Post.objects.all()
    serializer_class = PostSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def get_queryset(self):
        qs = super().get_queryset()
        # Optimize queries
        qs = qs.select_related('author')
        if self.action == 'list':
            qs = qs.filter(published=True)
        return qs

    def perform_create(self, serializer):
        serializer.save(author=self.request.user)

    @action(detail=True, methods=['post'])
    def publish(self, request, pk=None):
        post = self.get_object()
        post.published = True
        post.save()
        return Response({'status': 'published'})
```

## Serializers

```python
from rest_framework import serializers

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'email', 'username', 'bio']
        read_only_fields = ['id']


class PostSerializer(serializers.ModelSerializer):
    author = UserSerializer(read_only=True)

    class Meta:
        model = Post
        fields = ['id', 'author', 'title', 'content', 'published', 'created_at']
        read_only_fields = ['id', 'author', 'created_at']

    def validate_title(self, value):
        if len(value) < 5:
            raise serializers.ValidationError("Title too short")
        return value
```

## Query Optimization

```python
# Avoid N+1 queries
# BAD
posts = Post.objects.all()
for post in posts:
    print(post.author.username)  # N+1 queries!

# GOOD
posts = Post.objects.select_related('author').all()
for post in posts:
    print(post.author.username)  # 1 query

# For reverse relations (many)
users = User.objects.prefetch_related('posts').all()
for user in users:
    for post in user.posts.all():  # No extra queries
        print(post.title)
```

## Transactions

```python
from django.db import transaction

@transaction.atomic
def create_order(user, items):
    order = Order.objects.create(user=user)
    for item in items:
        OrderItem.objects.create(order=order, **item)
        update_inventory(item)  # Rollback if this fails
    return order

# Manual savepoints
def complex_operation():
    with transaction.atomic():
        do_something()
        sid = transaction.savepoint()
        try:
            risky_operation()
        except Exception:
            transaction.savepoint_rollback(sid)
        else:
            transaction.savepoint_commit(sid)
```

## Authentication (JWT)

```python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ],
}

# urls.py
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

urlpatterns = [
    path('api/token/', TokenObtainPairView.as_view()),
    path('api/token/refresh/', TokenRefreshView.as_view()),
]
```

## Best Practices

```
ALWAYS:
- Use select_related/prefetch_related
- Add database indexes for filtered fields
- Use transaction.atomic for critical operations
- Validate input in serializers

AVOID:
- Raw SQL unless necessary
- Fat views (move logic to services/managers)
- N+1 queries
- Storing secrets in settings.py
```

