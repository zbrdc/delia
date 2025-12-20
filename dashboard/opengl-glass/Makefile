CC = gcc
CFLAGS = $(shell pkg-config --cflags glfw3 glew)
LIBS = $(shell pkg-config --libs glfw3 glew) -lGL
TARGET = glass_window
SRC = glass_window.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(SRC) -o $(TARGET) $(CFLAGS) $(LIBS)

clean:
	rm -f $(TARGET)

run: all
	./$(TARGET)
