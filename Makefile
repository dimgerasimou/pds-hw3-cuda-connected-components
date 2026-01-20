PROJECT := connected_components_cuda

CC := gcc

CFLAGS  := -Wall -Wextra -Wpedantic -O3 -Isrc -fopenmp
LDFLAGS := -fopenmp -lm

# Directories
SRC_DIR := src
BIN_DIR := bin
OBJ_DIR := obj

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

CFLAGS += -MMD -MP
DEPS := $(OBJS:.o=.d)
-include $(DEPS)

TARGET := $(BIN_DIR)/$(PROJECT)

# Pretty Output
PRINTF        := printf
COLOR_RESET   := \033[0m
COLOR_BOLD    := \033[1m
COLOR_GREEN   := \033[1;32m
COLOR_YELLOW  := \033[1;33m
COLOR_BLUE    := \033[1;34m
COLOR_MAGENTA := \033[1;35m
COLOR_CYAN    := \033[1;36m

.PHONY: all
all: $(TARGET)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

$(TARGET): $(OBJS) | $(BIN_DIR)
	@$(PRINTF) "$(COLOR_GREEN)Linking:$(COLOR_RESET) $@\n"
	@$(CC) $(OBJS) -o $@ $(LDFLAGS) $(LDLIBS)
	@$(PRINTF) "$(COLOR_CYAN)Build complete!$(COLOR_RESET)\n"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	@$(PRINTF) "$(COLOR_BLUE)Compiling:$(COLOR_RESET) $<\n"
	@$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	@$(PRINTF) "$(COLOR_YELLOW)Cleaning...$(COLOR_RESET)\n"
	@rm -rf $(OBJ_DIR) $(BIN_DIR)
	@$(PRINTF) "$(COLOR_GREEN)✓ Clean complete$(COLOR_RESET)\n"

.PHONY: rebuild
rebuild: clean all

.PHONY: help
help:
	@$(PRINTF) "\n"
	@$(PRINTF) "$(COLOR_BOLD)$(COLOR_BLUE)Cuda Connected Components Benchmark$(COLOR_RESET)\n"
	@$(PRINTF) "\n"
	@$(PRINTF) "$(COLOR_BOLD)Targets:$(COLOR_RESET)\n"
	@$(PRINTF) "  $(COLOR_CYAN)all$(COLOR_RESET)       Build version (default)\n"
	@$(PRINTF) "  $(COLOR_CYAN)clean$(COLOR_RESET)     Remove build artifacts\n"
	@$(PRINTF) "  $(COLOR_CYAN)rebuild$(COLOR_RESET)   Clean and rebuild\n"
	@$(PRINTF) "  $(COLOR_CYAN)help$(COLOR_RESET)      Show this message\n"
	@$(PRINTF) "\n"
	@$(PRINTF) "$(COLOR_BOLD)Usage:$(COLOR_RESET)\n"
	@$(PRINTF) "  ./$(TARGET) [-n trials] [-w warmup trials] [-i implementation type] ./dest/to/data.mtx\n"
