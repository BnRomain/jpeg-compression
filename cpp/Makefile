# C++ version of the image compression project (DCT + CSR).
#
#   make                 builds the ./jpeg_csr program
#   make test            builds and runs the unit tests
#   make demo            compresses the sample image with several settings
#   make SANITIZE=1 ...  adds AddressSanitizer and UBSan (Linux, macOS)
#   make COVERAGE=1 ...  adds gcov instrumentation (report: gcovr --root . --filter src/)
#   make clean           removes the generated files

CXX = g++
CXXFLAGS ?= -std=c++20 -Wall -Wextra -pedantic -O2
# -isystem: stb is an external library, its warnings are not our concern.
CPPFLAGS += -Iinclude -isystem third_party

ifeq ($(SANITIZE),1)
CXXFLAGS += -g -fsanitize=address,undefined
LDFLAGS += -fsanitize=address,undefined
endif

# -O0 keeps the executed lines aligned with the source for the coverage report.
ifeq ($(COVERAGE),1)
CXXFLAGS += -O0 --coverage
LDFLAGS += --coverage
endif

# On Windows (MinGW), g++ adds the .exe extension and the program is linked
# statically: the executable no longer depends on the MinGW DLLs, which other
# versions found in the PATH (Git, MATLAB...) can shadow.
ifeq ($(OS),Windows_NT)
EXE := .exe
LDFLAGS += -static
endif

TARGET := jpeg_csr$(EXE)
TEST_TARGET := run_tests$(EXE)
BUILD := build

HEADERS := $(wildcard include/*.hpp)
LIB_SOURCES := $(filter-out src/main.cpp,$(wildcard src/*.cpp))
LIB_OBJECTS := $(patsubst src/%.cpp,$(BUILD)/%.o,$(LIB_SOURCES))

.PHONY: all test demo clean

all: $(TARGET)

$(TARGET): $(LIB_OBJECTS) $(BUILD)/main.o
	$(CXX) $(LDFLAGS) $^ -o $@

$(TEST_TARGET): $(LIB_OBJECTS) $(BUILD)/test_main.o
	$(CXX) $(LDFLAGS) $^ -o $@

# Every object is rebuilt as soon as a header changes: simple and good enough here.
$(BUILD)/%.o: src/%.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD)/test_main.o: tests/test_main.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $(BUILD)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

demo: $(TARGET)
	./$(TARGET) compress images/astronaut.png --out results/standard
	./$(TARGET) compress images/astronaut.png --alpha 5 --out results/alpha5
	./$(TARGET) compress images/astronaut.png --alpha 20 --out results/alpha20
	./$(TARGET) compress images/astronaut.png --noise 0.05 --mask triangle --cutoff 4 --out results/noise
	./$(TARGET) decompress results/standard/astronaut.csr results/standard/decoded.png

clean:
	rm -rf $(BUILD) results jpeg_csr jpeg_csr.exe run_tests run_tests.exe
