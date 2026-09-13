# Version C++ du projet de compression d'image (DCT + CSR).
#
#   make                 construit le programme ./jpeg_csr
#   make test            construit et exécute les tests unitaires
#   make demo            compresse l'image de test avec plusieurs réglages
#   make SANITIZE=1 ...  ajoute AddressSanitizer et UBSan (Linux, macOS)
#   make clean           supprime les fichiers produits

CXX = g++
CXXFLAGS ?= -std=c++20 -Wall -Wextra -pedantic -O2
# -isystem : stb est une bibliothèque externe, ses avertissements ne nous concernent pas.
CPPFLAGS += -Iinclude -isystem third_party

ifeq ($(SANITIZE),1)
CXXFLAGS += -g -fsanitize=address,undefined
LDFLAGS += -fsanitize=address,undefined
endif

# Sous Windows (MinGW), g++ ajoute l'extension .exe et l'on lie statiquement :
# l'exécutable ne dépend plus des DLL de MinGW, que d'autres versions présentes
# dans le PATH (Git, MATLAB...) peuvent masquer.
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

# Tout objet est recompilé dès qu'un en-tête change : simple et suffisant ici.
$(BUILD)/%.o: src/%.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD)/test_main.o: tests/test_main.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $(BUILD)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

demo: $(TARGET)
	./$(TARGET) compress images/astronaut.png --out resultats/standard
	./$(TARGET) compress images/astronaut.png --alpha 5 --out resultats/alpha5
	./$(TARGET) compress images/astronaut.png --alpha 20 --out resultats/alpha20
	./$(TARGET) compress images/astronaut.png --noise 0.05 --mask triangle --cutoff 4 --out resultats/bruit
	./$(TARGET) decompress resultats/standard/astronaut.csr resultats/standard/relue.png

clean:
	rm -rf $(BUILD) resultats jpeg_csr jpeg_csr.exe run_tests run_tests.exe
