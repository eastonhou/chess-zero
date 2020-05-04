BASE = $(shell pwd)
TORCH_API = -I$(BASE)/libtorch/include/torch/csrc/api/include
TORCH_INC = -I$(BASE)/libtorch/include
TORCH_LIB = -L$(BASE)/libtorch/lib
OUTPUT = $(BASE)/bin
BUILD = $(BASE)/build
OBJS = $(BASE)/build/objs
CFLAGS = -Wall -Werror -std=c++14 -g -O
prebuild:
	mkdir $(OUTPUT)
	mkdir -p $(OBJS)

train: train.o
	g++ $(CFLAGS) -o $(OUTPUT)/$@ $(OBJS)/$^ $(TORCH_LIB)

train.o: clean prebuild
	g++ $(CFLAGS) $(TORCH_API) $(TORCH_INC) -c train.cpp -o $(OBJS)/train.o

clean:
	rm -rf qwirkle $(OUTPUT) $(build)
