BASE = $(shell pwd)
ifeq ($(build), debug)
TORCH_BASE = libtorch/debug
CFLAGS = -Wall -Werror -std=c++17 -g -Ddebug
else
TORCH_BASE = libtorch/release
CFLAGS = -Wall -Werror -std=c++17 -g -O3
endif

TORCH_API = -I$(BASE)/$(TORCH_BASE)/include/torch/csrc/api/include
TORCH_INC = -I$(BASE)/$(TORCH_BASE)/include
TORCH_LIB = -L$(BASE)/$(TORCH_BASE)/lib
OUTPUT = $(BASE)/bin
BUILD = $(BASE)/build
OBJS = $(BASE)/build/objs

prebuild:
	mkdir $(OUTPUT)
	mkdir -p $(OBJS)

train: train.o
	g++ $(CFLAGS) -o $(OUTPUT)/$@ $(OBJS)/$^ $(TORCH_LIB)

train.o: clean prebuild
	g++ $(CFLAGS) $(TORCH_API) $(TORCH_INC) -c train.cpp -o $(OBJS)/train.o

clean:
	rm -rf qwirkle $(OUTPUT) $(build)
