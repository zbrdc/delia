# Rust Profile

Load this profile for: Rust applications, systems programming, async, memory safety.

## Project Structure

```
project/
├── src/
│   ├── main.rs              # Binary entry
│   ├── lib.rs               # Library root
│   ├── config.rs
│   ├── error.rs
│   ├── models/
│   │   └── mod.rs
│   └── handlers/
│       └── mod.rs
├── tests/                   # Integration tests
├── benches/                 # Benchmarks
├── Cargo.toml
└── Cargo.lock
```

## Error Handling

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("User not found: {0}")]
    NotFound(String),

    #[error("Invalid input: {0}")]
    Validation(String),

    #[error("Database error")]
    Database(#[from] sqlx::Error),

    #[error("IO error")]
    Io(#[from] std::io::Error),
}

// Use Result type alias
pub type Result<T> = std::result::Result<T, AppError>;

// Propagate with ?
async fn get_user(id: &str) -> Result<User> {
    let user = db.fetch_one(id).await?;
    Ok(user)
}
```

## Ownership & Borrowing

```rust
// Take ownership when you need to own
fn consume(s: String) { /* owns s */ }

// Borrow when you just need to read
fn read(s: &str) { /* borrows s */ }

// Mutable borrow when you need to modify
fn modify(s: &mut String) { s.push_str("!"); }

// Common patterns
impl User {
    // Take &self for read methods
    pub fn name(&self) -> &str {
        &self.name
    }

    // Take &mut self for mutations
    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    // Take self for consuming operations
    pub fn into_dto(self) -> UserDto {
        UserDto { name: self.name }
    }
}
```

## Async/Tokio Patterns

```rust
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    // Spawn tasks
    let handle = tokio::spawn(async {
        expensive_operation().await
    });

    // Parallel execution
    let (a, b, c) = tokio::join!(
        fetch_users(),
        fetch_orders(),
        fetch_settings(),
    );

    // Select first completed
    tokio::select! {
        result = operation1() => handle_result1(result),
        result = operation2() => handle_result2(result),
        _ = tokio::time::sleep(Duration::from_secs(5)) => {
            println!("Timeout!");
        }
    }
}

// Channel communication
async fn worker(mut rx: mpsc::Receiver<Job>) {
    while let Some(job) = rx.recv().await {
        process(job).await;
    }
}
```

## Traits & Generics

```rust
// Define traits for abstraction
pub trait Repository {
    type Entity;
    type Error;

    async fn find(&self, id: &str) -> Result<Self::Entity, Self::Error>;
    async fn save(&self, entity: &Self::Entity) -> Result<(), Self::Error>;
}

// Generic functions
fn process<T: Display + Clone>(items: &[T]) -> Vec<String> {
    items.iter().map(|i| i.to_string()).collect()
}

// Trait bounds with where
fn complex_operation<T, U>(t: T, u: U) -> String
where
    T: Display + Clone,
    U: Debug + Send,
{
    format!("{} {:?}", t, u)
}
```

## Option & Result Combinators

```rust
// Option chaining
let name = user
    .profile
    .as_ref()
    .and_then(|p| p.display_name.as_ref())
    .unwrap_or(&user.username);

// Result transformation
let user: User = response
    .json()
    .await
    .map_err(|e| AppError::Parse(e.to_string()))?;

// Collect Results
let users: Result<Vec<User>> = ids
    .iter()
    .map(|id| fetch_user(id))
    .collect();
```

## Best Practices

```
ALWAYS:
- Use clippy and rustfmt
- Handle all Result/Option cases
- Prefer &str over String for parameters
- Use #[derive] for common traits
- Document public APIs with ///

AVOID:
- .unwrap() in production code
- Clone without justification
- Unsafe without clear need
- Blocking in async contexts
```

