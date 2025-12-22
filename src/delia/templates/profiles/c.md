# C Profile

Load this profile for: C programming, embedded systems, low-level development.

## Project Structure

```
project/
├── src/
│   ├── main.c
│   ├── core/
│   │   ├── engine.c
│   │   └── engine.h
│   └── utils/
│       ├── memory.c
│       └── memory.h
├── include/
│   └── project/
│       └── public_api.h
├── tests/
├── Makefile
└── CMakeLists.txt
```

## Header Guards & Includes

```c
// engine.h
#ifndef PROJECT_ENGINE_H
#define PROJECT_ENGINE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Declarations here

#ifdef __cplusplus
}
#endif

#endif // PROJECT_ENGINE_H
```

## Memory Management

```c
#include <stdlib.h>
#include <string.h>

// Always check allocation
void* safe_malloc(size_t size) {
    void* ptr = malloc(size);
    if (ptr == NULL && size > 0) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// Zero-initialize
void* safe_calloc(size_t count, size_t size) {
    void* ptr = calloc(count, size);
    if (ptr == NULL && count > 0 && size > 0) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// Realloc pattern
int resize_buffer(Buffer* buf, size_t new_size) {
    void* new_data = realloc(buf->data, new_size);
    if (new_data == NULL && new_size > 0) {
        return -1;  // Keep old buffer intact
    }
    buf->data = new_data;
    buf->capacity = new_size;
    return 0;
}

// Cleanup pattern
void buffer_free(Buffer* buf) {
    if (buf) {
        free(buf->data);
        buf->data = NULL;
        buf->size = 0;
        buf->capacity = 0;
    }
}
```

## Error Handling

```c
#include <errno.h>

// Return codes
typedef enum {
    ERR_OK = 0,
    ERR_NULL_PTR = -1,
    ERR_OUT_OF_MEMORY = -2,
    ERR_INVALID_ARG = -3,
    ERR_IO = -4,
} ErrorCode;

// Function with error return
ErrorCode read_file(const char* path, Buffer* out) {
    if (path == NULL || out == NULL) {
        return ERR_NULL_PTR;
    }

    FILE* fp = fopen(path, "rb");
    if (fp == NULL) {
        return ERR_IO;
    }

    // Get file size
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    // Allocate buffer
    out->data = malloc(size);
    if (out->data == NULL) {
        fclose(fp);
        return ERR_OUT_OF_MEMORY;
    }

    // Read file
    size_t read = fread(out->data, 1, size, fp);
    fclose(fp);

    if (read != (size_t)size) {
        free(out->data);
        out->data = NULL;
        return ERR_IO;
    }

    out->size = size;
    return ERR_OK;
}

// Error to string
const char* error_string(ErrorCode err) {
    switch (err) {
        case ERR_OK: return "Success";
        case ERR_NULL_PTR: return "Null pointer";
        case ERR_OUT_OF_MEMORY: return "Out of memory";
        case ERR_INVALID_ARG: return "Invalid argument";
        case ERR_IO: return "I/O error";
        default: return "Unknown error";
    }
}
```

## Data Structures

```c
// Dynamic array
typedef struct {
    int* data;
    size_t size;
    size_t capacity;
} IntArray;

int int_array_init(IntArray* arr, size_t initial_capacity) {
    arr->data = malloc(initial_capacity * sizeof(int));
    if (arr->data == NULL) return -1;
    arr->size = 0;
    arr->capacity = initial_capacity;
    return 0;
}

int int_array_push(IntArray* arr, int value) {
    if (arr->size >= arr->capacity) {
        size_t new_cap = arr->capacity * 2;
        int* new_data = realloc(arr->data, new_cap * sizeof(int));
        if (new_data == NULL) return -1;
        arr->data = new_data;
        arr->capacity = new_cap;
    }
    arr->data[arr->size++] = value;
    return 0;
}

void int_array_free(IntArray* arr) {
    free(arr->data);
    arr->data = NULL;
    arr->size = 0;
    arr->capacity = 0;
}
```

## String Handling

```c
#include <string.h>

// Safe string copy
size_t safe_strcpy(char* dest, size_t dest_size, const char* src) {
    if (dest == NULL || dest_size == 0) return 0;
    if (src == NULL) {
        dest[0] = '\0';
        return 0;
    }

    size_t src_len = strlen(src);
    size_t copy_len = (src_len < dest_size - 1) ? src_len : dest_size - 1;

    memcpy(dest, src, copy_len);
    dest[copy_len] = '\0';

    return copy_len;
}

// String builder
typedef struct {
    char* data;
    size_t len;
    size_t capacity;
} StringBuilder;

int sb_append(StringBuilder* sb, const char* str) {
    size_t str_len = strlen(str);
    size_t new_len = sb->len + str_len;

    if (new_len >= sb->capacity) {
        size_t new_cap = (new_len + 1) * 2;
        char* new_data = realloc(sb->data, new_cap);
        if (new_data == NULL) return -1;
        sb->data = new_data;
        sb->capacity = new_cap;
    }

    memcpy(sb->data + sb->len, str, str_len + 1);
    sb->len = new_len;
    return 0;
}
```

## Macros

```c
// Utility macros
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLAMP(x, lo, hi) MIN(MAX(x, lo), hi)

// Container of
#define container_of(ptr, type, member) \
    ((type *)((char *)(ptr) - offsetof(type, member)))

// Likely/unlikely for branch prediction
#if defined(__GNUC__)
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x)   (x)
#define unlikely(x) (x)
#endif
```

## Makefile Template

```makefile
CC = gcc
CFLAGS = -Wall -Wextra -Wpedantic -std=c11 -O2
LDFLAGS =
DEBUG_FLAGS = -g -O0 -DDEBUG

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

SRCS = $(wildcard $(SRC_DIR)/*.c $(SRC_DIR)/**/*.c)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
TARGET = $(BIN_DIR)/program

.PHONY: all clean debug

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(LDFLAGS) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

debug: CFLAGS += $(DEBUG_FLAGS)
debug: clean all

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
```

## Best Practices

```
ALWAYS:
- Check all allocation returns
- Free in reverse order of allocation
- Use const for read-only parameters
- Validate all input parameters
- Initialize variables at declaration

AVOID:
- Buffer overflows (use bounded functions)
- Use-after-free
- Double free
- Uninitialized variables
- Magic numbers (use defines/enums)
```

## Safety Checklist

```
□ All malloc/calloc/realloc checked for NULL
□ All array accesses bounds-checked
□ All strings null-terminated
□ No buffer overflows in string operations
□ No memory leaks (valgrind clean)
□ No undefined behavior (UBSan clean)
```

