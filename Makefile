# ─────────────────────────────────────────────────────────────
#  Makefile — Salmon Chess Engine
#
#  Targets:
#    make              — release build  (bin/engine)
#    make debug        — debug build with symbols
#    make sanitize     — ASan + UBSan build
#    make tuner        — build the tuner (bin/tuna)
#    make perft        — run a quick built-in perft test
#    make clean        — remove build artefacts
#
#  The -march=native flag lets GCC/Clang automatically enable
#  all ISA extensions the host CPU supports (POPCNT, BMI2,
#  AVX2, AVX-512, etc.).  The bitboard and search code uses
#  __builtin_popcountll, __builtin_ctzll, __builtin_bswap64
#  which are always available; AVX2 / AVX-512 paths are gated
#  on the preprocessor defines GCC emits for -march=native.
# ─────────────────────────────────────────────────────────────

CC      := gcc
TARGET  := bin/engine
TUNER   := bin/tuna

SRC_DIR := src
INC_DIR := include
BIN_DIR := bin
OBJ_DIR := build

SRCS := $(filter-out $(SRC_DIR)/tune.c, $(wildcard $(SRC_DIR)/*.c))
OBJS := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

TUNER_SRCS := $(SRC_DIR)/tune.c $(SRC_DIR)/bitboard.c $(SRC_DIR)/board.c $(SRC_DIR)/movegen.c
TUNER_OBJS := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(TUNER_SRCS))

# ── Common flags ─────────────────────────────────────────────
CFLAGS_COMMON := \
    -std=c2x \
    -I$(INC_DIR) \
    -Wall -Wextra -Wpedantic \
    -Wshadow -Wconversion -Wno-unused-parameter

# ── Release flags ────────────────────────────────────────────
CFLAGS_RELEASE := \
    $(CFLAGS_COMMON) \
    -O3 \
    -march=native \
    -DNDEBUG \
    -fomit-frame-pointer \
    -flto

LDFLAGS_RELEASE := -flto -lm

# ── Debug flags ──────────────────────────────────────────────
CFLAGS_DEBUG := \
    $(CFLAGS_COMMON) \
    -O0 -g3 \
    -march=native \
    -DENGINE_DEBUG

LDFLAGS_DEBUG := -lm

# ── Sanitize flags ───────────────────────────────────────────
CFLAGS_SANITIZE := \
    $(CFLAGS_COMMON) \
    -O1 -g \
    -fsanitize=address,undefined \
    -fno-omit-frame-pointer

LDFLAGS_SANITIZE := -fsanitize=address,undefined -lm

# ── Default (release) ────────────────────────────────────────
CFLAGS  := $(CFLAGS_RELEASE)
LDFLAGS := $(LDFLAGS_RELEASE)

# ─────────────────────────────────────────────────────────────
.PHONY: all debug sanitize tuner perft clean

all: $(BIN_DIR)/$(notdir $(TARGET))

# Release
$(BIN_DIR)/$(notdir $(TARGET)): $(OBJS) | $(BIN_DIR)
	$(CC) $(LDFLAGS) -o $@ $^
	@echo "Built: $@"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug
debug: CFLAGS  := $(CFLAGS_DEBUG)
debug: LDFLAGS := $(LDFLAGS_DEBUG)
debug: TARGET  := $(BIN_DIR)/engine_debug
debug: $(BIN_DIR)/engine_debug

$(BIN_DIR)/engine_debug: $(SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS_DEBUG) $(LDFLAGS_DEBUG) -o $@ $^
	@echo "Built debug: $@"

# Sanitize
sanitize: $(SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS_SANITIZE) $(LDFLAGS_SANITIZE) -o $(BIN_DIR)/engine_san $^
	@echo "Built sanitized: $(BIN_DIR)/engine_san"

# Tuner
tuner: CFLAGS := $(CFLAGS_RELEASE) -DTUNE_STANDALONE
tuner: LDFLAGS := $(LDFLAGS_RELEASE)
tuner: $(BIN_DIR)/tuna

$(BIN_DIR)/tuna: $(TUNER_OBJS) | $(BIN_DIR)
	$(CC) $(LDFLAGS) -o $@ $^
	@echo "Built tuner: $@"

tuner_debug: CFLAGS := $(CFLAGS_DEBUG) -DTUNE_STANDALONE
tuner_debug: LDFLAGS := $(LDFLAGS_DEBUG)
tuner_debug: TARGET  := $(BIN_DIR)/tuna_debug
tuner_debug: $(BIN_DIR)/tuna_debug

$(BIN_DIR)/tuna_debug: $(TUNER_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS_DEBUG) -DTUNE_STANDALONE $(LDFLAGS_DEBUG) -o $@ $^
	@echo "Built debug tuna: $@"

# Create directories
$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

# Quick perft smoke test (position 5 from CPW, depth 3, expected 62379)
perft: all
	@echo "Running perft depth 3 on position 5..."
	@echo -e "position fen rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8\nperft 3\nquit" \
	    | ./$(BIN_DIR)/$(notdir $(TARGET))

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "Cleaned."
