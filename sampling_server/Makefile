default: build

# Directories
HERE := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
ROOT := $(realpath $(HERE))
BUILD_DIR := $(ROOT)/build
SRC_DIR := $(ROOT)/src
INCLUDE_DIR := $(SRC_DIR)/include
STORAGE_DIR := $(SRC_DIR)/storage
CACHE_DIR := $(SRC_DIR)/cache
ENGINE_DIR := $(SRC_DIR)/engine

# Output library name
PYTHON_PACKAGE_NAME := hyperion
LIB_OUTPUT := $(BUILD_DIR)/$(PYTHON_PACKAGE_NAME).so

# Compiler and flags
NVCXX := nvcc
NVCXXFLAGS := -std=c++17 -arch=sm_60 -Xcompiler -Wall \
	--extended-lambda \
	-Xcompiler -fPIC -lpthread \
	--expt-relaxed-constexpr \
	-I $(INCLUDE_DIR) \
	-I $(INCLUDE_DIR)/hashmap \
	-I $(STORAGE_DIR) \
	-I $(CACHE_DIR) \
	-I $(ENGINE_DIR) \
	-g

CXX := g++
CXXFLAGS := -std=c++17 -fPIC -O2 -g \
	-I $(INCLUDE_DIR) \
	-I $(STORAGE_DIR) \
	-I $(CACHE_DIR) \
	-I $(ENGINE_DIR) \
	-I $(INCLUDE_DIR)/hashmap

# Pybind11
PYBIND11_INCLUDE := $(shell python3 -m pybind11 --includes)

# Source files
CUDA_FILES := $(wildcard $(SRC_DIR)/**/*.cu)
CPP_FILES := $(wildcard $(SRC_DIR)/**/*.cpp)
HYPERION_CPP := hyperion.cpp
CUDA_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CUDA_FILES))
CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_FILES))
HYPERION_OBJ := $(BUILD_DIR)/$(HYPERION_CPP:.cpp=.o)

# Clean build artifacts
clean:
	@rm -rf $(BUILD_DIR)

# Compile CUDA source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCXX) $(NVCXXFLAGS) -c $< -o $@

# Compile C++ source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDE) -c $< -o $@

# Compile Pybind11 entry file
$(HYPERION_OBJ): $(HYPERION_CPP)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDE) -c $< -o $@

# Build Python module
$(LIB_OUTPUT): $(CUDA_OBJS) $(CPP_OBJS) $(HYPERION_OBJ)
	$(NVCXX) $(NVCXXFLAGS) -shared -o $@ $^ -lrt

build: $(LIB_OUTPUT)

.PHONY: clean build
