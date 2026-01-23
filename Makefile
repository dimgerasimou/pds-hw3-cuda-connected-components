PROJECT ?= connected_components_cuda

CC   ?= gcc

# ---------- Directories ----------
SRC_DIR ?= src
OBJ_DIR ?= obj
BIN_DIR ?= bin

# ---------- CUDA location ----------
# Try to auto-detect CUDA from nvcc in PATH first, then fall back to common locations
ifndef CUDA_HOME
  NVCC_PATH := $(shell which nvcc 2>/dev/null)
  ifneq ($(NVCC_PATH),)
    CUDA_HOME := $(shell dirname $(shell dirname $(NVCC_PATH)))
  else
    CUDA_HOME := $(firstword \
      $(wildcard /opt/cuda) \
      $(wildcard /usr/local/cuda) \
    )
  endif
endif

# If neither exists, CUDA_HOME stays empty; nvcc can still work if it knows its install.
CUDA_INC ?= $(if $(CUDA_HOME),$(CUDA_HOME)/include,)

CUDA_LIB ?= $(if $(CUDA_HOME),$(CUDA_HOME)/lib64,)

NVCC ?= $(if $(CUDA_HOME),$(CUDA_HOME)/bin/nvcc,nvcc)

# ---------- CUDA architecture ----------
#
# Why this exists:
# Newer nvcc can emit PTX that an older driver cannot JIT (e.g., nvcc 12.5
# with a driver that supports CUDA 12.4). To avoid "PTX was compiled with an
# unsupported toolchain" at kernel launch, we compile native SASS (sm_XX)
# for the detected/selected GPU.
#
# Usage:
#   make                  # auto-detect GPU, fallback to T4 (sm_75)
#   make GPU_ARCH=86      # override (e.g., RTX 3060)
#   make GPU_ARCHES="75 86"  # build fat binary for multiple GPUs

# Auto-detect compute capability using nvidia-smi (first GPU), if available.
# Output is typically "7.5"; convert to "75".
GPU_ARCH_DETECTED := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.')

# Default arch:
# - If we can detect one, use it.
# - Otherwise default to sm_75 (Colab T4 baseline).
GPU_ARCH ?= $(if $(GPU_ARCH_DETECTED),$(GPU_ARCH_DETECTED),75)

# Optional multi-arch build. If unset, we compile only for GPU_ARCH.
GPU_ARCHES ?= $(GPU_ARCH)

NVCC_GENCODE :=
$(foreach arch,$(GPU_ARCHES),$(eval NVCC_GENCODE += -gencode arch=compute_$(arch),code=sm_$(arch)))

# ---------- Flags ----------
CPPFLAGS  ?=
CFLAGS    ?= -Wall -Wextra -Wpedantic -O3 -fopenmp
NVCCFLAGS ?= -O3 -Xcompiler=-fopenmp --use_fast_math

# Emit native SASS for the selected arch(es) to avoid PTX JIT/toolchain issues.
NVCCFLAGS += $(NVCC_GENCODE)

# Print the effective build target(s) once (useful on Colab/cluster).
# $(info [1;35m[CUDA] build arch(es):[0m $(GPU_ARCHES) (default GPU_ARCH=$(GPU_ARCH), detected=$(GPU_ARCH_DETECTED)))

LDFLAGS   ?= -Xcompiler=-fopenmp
LDLIBS    ?= -lm -lcudart

# Add include paths
CPPFLAGS += -I$(SRC_DIR)
ifneq ($(strip $(CUDA_INC)),)
CPPFLAGS += -I$(CUDA_INC)
endif

# Add library search path
ifneq ($(strip $(CUDA_LIB)),)
LDFLAGS += -L$(CUDA_LIB)
endif

# Dependency generation
DEPFLAGS := -MMD -MP

# ---------- Sources ----------
# Adjust this pattern if you meant to exclude specific files:
C_SRCS  := $(filter-out $(SRC_DIR)/cc_cuda%.c, $(wildcard $(SRC_DIR)/*.c))
CU_SRCS := $(wildcard $(SRC_DIR)/*.cu)

C_OBJS  := $(C_SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
CU_OBJS := $(CU_SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
OBJS    := $(C_OBJS) $(CU_OBJS)

TARGET  ?= bin/$(PROJECT)

DEPS := $(OBJS:.o=.d)
-include $(DEPS)

# ---------- Pretty output (optional colors) ----------
PRINTF ?= printf
ifeq ($(NO_COLOR),1)
  COLOR_RESET   :=
  COLOR_BOLD    :=
  COLOR_GREEN   :=
  COLOR_YELLOW  :=
  COLOR_BLUE    :=
  COLOR_MAGENTA :=
  COLOR_CYAN    :=
else
  COLOR_RESET   := \033[0m
  COLOR_BOLD    := \033[1m
  COLOR_GREEN   := \033[1;32m
  COLOR_YELLOW  := \033[1;33m
  COLOR_BLUE    := \033[1;34m
  COLOR_MAGENTA := \033[1;35m
  COLOR_CYAN    := \033[1;36m
endif

# ---------- Rules ----------
.PHONY: all clean rebuild help
all: $(TARGET)

$(BIN_DIR) $(OBJ_DIR):
	@mkdir -p $@

# Link with nvcc so CUDA runtime + host compiler flags are handled sanely.
$(TARGET): $(OBJS) | $(BIN_DIR)
	@$(PRINTF) "$(COLOR_GREEN)Linking:$(COLOR_RESET) %s\n" "$@"
	@$(NVCC) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS)
	@$(PRINTF) "$(COLOR_CYAN)Build complete!$(COLOR_RESET)\n"

# C objects
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	@$(PRINTF) "$(COLOR_BLUE)Compiling C:$(COLOR_RESET) %s\n" "$<"
	@$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

# CUDA objects (generate .d next to .o; nvcc supports -MMD/-MP on modern toolkits)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	@$(PRINTF) "$(COLOR_MAGENTA)Compiling CUDA [Arch:$(GPU_ARCHES)]:$(COLOR_RESET) %s\n" "$<"
	@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(DEPFLAGS) -c $< -o $@

clean:
	@$(PRINTF) "$(COLOR_YELLOW)Cleaning...$(COLOR_RESET)\n"
	@rm -rf $(OBJ_DIR) $(BIN_DIR)
	@$(PRINTF) "$(COLOR_GREEN)✓ Clean complete$(COLOR_RESET)\n"

rebuild: clean all

help:
	@$(PRINTF) "\n"
	@$(PRINTF) "$(COLOR_BOLD)$(COLOR_BLUE)CUDA Connected Components Benchmark$(COLOR_RESET)\n\n"
	@$(PRINTF) "$(COLOR_BOLD)Targets:$(COLOR_RESET)\n"
	@$(PRINTF) "  $(COLOR_CYAN)all$(COLOR_RESET)       Build (default)\n"
	@$(PRINTF) "  $(COLOR_CYAN)clean$(COLOR_RESET)     Remove build artifacts\n"
	@$(PRINTF) "  $(COLOR_CYAN)rebuild$(COLOR_RESET)   Clean and rebuild\n"
	@$(PRINTF) "  $(COLOR_CYAN)help$(COLOR_RESET)      Show this message\n\n"
	@$(PRINTF) "$(COLOR_BOLD)Overrides:$(COLOR_RESET)\n"
	@$(PRINTF) "  make CUDA_HOME=/opt/cuda\n"
	@$(PRINTF) "  make GPU_ARCH=86\n"
	@$(PRINTF) "  make GPU_ARCHES=\"75 86 89\"\n"
	@$(PRINTF) "  make CC=clang\n"
	@$(PRINTF) "  make NO_COLOR=1\n"
	@$(PRINTF) "$(COLOR_BOLD)Usage:$(COLOR_RESET)\n"
	@$(PRINTF) "  ./$(TARGET) [-n trials] [-w warmup trials] [-i implementation type] data.mtx\n\n"

