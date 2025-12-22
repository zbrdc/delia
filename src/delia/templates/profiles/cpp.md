# C++ Profile

Load this profile for: Modern C++ (17/20), systems programming, performance-critical code.

## Project Structure

```
project/
├── src/
│   ├── main.cpp
│   ├── core/
│   │   ├── engine.cpp
│   │   └── engine.hpp
│   └── utils/
├── include/
│   └── project/
│       └── public_api.hpp
├── tests/
│   └── test_engine.cpp
├── benchmarks/
├── CMakeLists.txt
└── conanfile.txt
```

## Modern C++ Patterns

```cpp
#include <memory>
#include <optional>
#include <string_view>
#include <span>

// Use smart pointers
class ResourceManager {
public:
    std::unique_ptr<Resource> createResource() {
        return std::make_unique<Resource>();
    }

    std::shared_ptr<Cache> getCache() {
        return cache_;
    }

private:
    std::shared_ptr<Cache> cache_;
};

// Use std::optional for nullable returns
std::optional<User> findUser(std::string_view id) {
    auto it = users_.find(std::string(id));
    if (it == users_.end()) {
        return std::nullopt;
    }
    return it->second;
}

// Use std::string_view for read-only strings
void processName(std::string_view name) {
    // No copy, just a view
    std::cout << name << '\n';
}

// Use std::span for array views (C++20)
void processData(std::span<const int> data) {
    for (int val : data) {
        // Process without copying
    }
}
```

## RAII & Resource Management

```cpp
// RAII wrapper
class FileHandle {
public:
    explicit FileHandle(const char* path)
        : handle_(fopen(path, "r")) {
        if (!handle_) {
            throw std::runtime_error("Failed to open file");
        }
    }

    ~FileHandle() {
        if (handle_) {
            fclose(handle_);
        }
    }

    // Delete copy
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;

    // Enable move
    FileHandle(FileHandle&& other) noexcept
        : handle_(std::exchange(other.handle_, nullptr)) {}

    FileHandle& operator=(FileHandle&& other) noexcept {
        if (this != &other) {
            if (handle_) fclose(handle_);
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    FILE* get() const { return handle_; }

private:
    FILE* handle_;
};
```

## Error Handling

```cpp
#include <expected>  // C++23
#include <system_error>
#include <variant>

// Using std::expected (C++23)
std::expected<User, std::error_code> getUser(int id) {
    auto result = db_.query(id);
    if (!result) {
        return std::unexpected(make_error_code(std::errc::no_such_file_or_directory));
    }
    return *result;
}

// Pre-C++23: Use variant or custom Result type
template<typename T, typename E = std::string>
class Result {
public:
    static Result ok(T value) { return Result(std::move(value)); }
    static Result err(E error) { return Result(std::move(error), false); }

    bool isOk() const { return std::holds_alternative<T>(data_); }
    T& value() { return std::get<T>(data_); }
    E& error() { return std::get<E>(data_); }

private:
    std::variant<T, E> data_;
    Result(T value) : data_(std::move(value)) {}
    Result(E error, bool) : data_(std::move(error)) {}
};
```

## Concurrency

```cpp
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>

class ThreadPool {
public:
    explicit ThreadPool(size_t numThreads) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers_.emplace_back([this] { workerLoop(); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& worker : workers_) {
            worker.join();
        }
    }

    template<typename F>
    auto submit(F&& task) -> std::future<std::invoke_result_t<F>> {
        using ReturnType = std::invoke_result_t<F>;
        auto packagedTask = std::make_shared<std::packaged_task<ReturnType()>>(
            std::forward<F>(task)
        );
        auto future = packagedTask->get_future();

        {
            std::lock_guard lock(mutex_);
            tasks_.emplace([packagedTask] { (*packagedTask)(); });
        }
        cv_.notify_one();

        return future;
    }

private:
    void workerLoop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
};
```

## Templates & Concepts (C++20)

```cpp
#include <concepts>

// Concept definition
template<typename T>
concept Hashable = requires(T t) {
    { std::hash<T>{}(t) } -> std::convertible_to<std::size_t>;
};

// Constrained template
template<Hashable T>
class HashSet {
public:
    void insert(const T& value) {
        auto hash = std::hash<T>{}(value);
        // ...
    }
};

// Constrained auto
void process(Hashable auto&& item) {
    // ...
}
```

## Performance Best Practices

```cpp
// Move semantics
std::vector<std::string> createStrings() {
    std::vector<std::string> result;
    result.reserve(100);  // Avoid reallocations
    // ...
    return result;  // NRVO or move
}

// Avoid copies
for (const auto& item : container) {  // Reference, not copy
    process(item);
}

// Use emplace
container.emplace_back(arg1, arg2);  // Construct in-place

// Prefer algorithms
std::ranges::sort(vec);
auto it = std::ranges::find(vec, target);
```

## Best Practices

```
ALWAYS:
- Use smart pointers over raw pointers
- Apply RAII for resource management
- Enable move semantics for large objects
- Use const-correctness throughout
- Prefer standard library algorithms

AVOID:
- Raw new/delete (use smart pointers)
- C-style casts (use static_cast, etc.)
- Macros for constants (use constexpr)
- Raw arrays (use std::array/vector)
```

## CMake Template

```cmake
cmake_minimum_required(VERSION 3.20)
project(MyProject VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/core/engine.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Testing
enable_testing()
find_package(GTest REQUIRED)
add_executable(tests tests/test_engine.cpp)
target_link_libraries(tests GTest::gtest_main)
add_test(NAME tests COMMAND tests)
```

