default: main

HERE := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
ROOT := $(realpath $(HERE))
BUILD_DIR := $(ROOT)/build
INCLUDE_DIR := $(BUILD_DIR)/include
LIB_DIR := $(BUILD_DIR)/lib
BIN_DIR := $(BUILD_DIR)/bin

THIRD_PARTY_DIR := $(ROOT)/third_party
PYTHON_DIR := $(ROOT)/python
SRC_DIR = $(ROOT)/src
SSD_DRIVER_DIR := /share/gnn_data/bam/src/linux


clean:
	@rm -rf $(BUILD_DIR)

################################## compiler #######################################

NVCXX := nvcc
NVCXXFLAGS := -std=c++17 -arch=sm_60 -Xcompiler -Wall \
			  --extended-lambda \
			  -Xcompiler -fPIC -lpthread \
			  --expt-relaxed-constexpr \
			  -I $(SRC_DIR)/include \
			  -I $(SRC_DIR)/cache \
			  -I $(SRC_DIR)/engine \
			  -I $(SRC_DIR)/storage \
			  -I $(SRC_DIR)/include/hashmap \
			  -I $(SSD_DRIVER_DIR) -g


CXX := g++
CXXFLAGS := -std=c++17 -fPIC -O2 \
			-I $(SRC_DIR)/engine -g

################################## main ########################################### 
main: core
	@mkdir -p $(BUILD_DIR)/built
	@mkdir -p $(INCLUDE_DIR)
	@mkdir -p $(LIB_DIR)
	@mkdir -p $(BIN_DIR)
	@echo $(CORE_H)
	$(NVCXX) $(NVCXXFLAGS) \
		-L /usr/local/cuda-11.7/lib64 \
		$(CORE_OBJ) \
		-lcudart -lpthread\
		-o $(BIN_DIR)/server -lrt


################################## core ########################################### 
CORE_DIR := $(SRC_DIR)
CORE_BUILT_DIR := $(BUILD_DIR)/built
CORE_DIRS := $(shell find "src" -maxdepth 3 -type d)
CORE_H := $(foreach dir,$(CORE_DIRS),$(wildcard $(dir)/*.h))
CORE_CUH := $(foreach dir,$(CORE_DIRS),$(wildcard $(dir)/*.cuh))
CORE_CU := $(foreach dir,$(CORE_DIRS),$(wildcard $(dir)/*.cu))
CORE_OBJ := $(addprefix $(CORE_BUILT_DIR)/,$(patsubst %.cu,%.o,$(CORE_CU)))

$(CORE_BUILT_DIR)/src/%.o:$(CORE_DIR)/%.cu $(CORE_H) $(CORE_CUH)
	@mkdir -p $(CORE_BUILT_DIR)
	@mkdir -p $(CORE_BUILT_DIR)/src
	$(NVCXX) $(NVCXXFLAGS) -c $< -o $@ -lrt
	
$(CORE_BUILT_DIR)/src/cache/%.o:$(CORE_DIR)/cache/%.cu $(CORE_H) $(CORE_CUH)
	@mkdir -p $(CORE_BUILT_DIR)/src/cache
	@echo $<
	$(NVCXX) $(NVCXXFLAGS) -c $< -o $@ -lrt

$(CORE_BUILT_DIR)/src/storage/%.o:$(CORE_DIR)/storage/%.cu $(CORE_H) $(CORE_CUH)
	@mkdir -p $(CORE_BUILT_DIR)/src/storage
	$(NVCXX) $(NVCXXFLAGS) -c $< -o $@ -lrt

$(CORE_BUILT_DIR)/src/engine/%.o:$(CORE_DIR)/engine/%.cu $(CORE_H) $(CORE_CUH)
	@mkdir -p $(CORE_BUILT_DIR)/src/engine
	$(NVCXX) $(NVCXXFLAGS) -c $< -o $@ -lrt

core: $(CORE_OBJ)
