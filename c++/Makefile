BASE = $(shell pwd)
ifeq ($(build), debug)
TORCH_BASE = libtorch/release
CFLAGS = -Wall -Werror -m64 -std=c++17 -g -Ddebug -D_GLIBCXX_USE_CXX11_ABI=0 -D_LIBCPP_DEBUG
TORCH_LIBS = -ltorch -lc10 -lcaffe2_nvrtc -ltorch_cpu -ltorch_cuda -lc10_cuda
STDLIBS = -lstdc++fs -lpthread
else
TORCH_BASE = libtorch/release
CFLAGS = -Wall -Werror -m64 -std=c++17 -g -O3 -D_GLIBCXX_USE_CXX11_ABI=0
TORCH_LIBS = -ltorch -lc10 -lcaffe2_nvrtc -ltorch_cpu -ltorch_cuda -lc10_cuda
STDLIBS = -lstdc++fs -lpthread
endif

TORCH_API = -I$(BASE)/$(TORCH_BASE)/include/torch/csrc/api/include
TORCH_INC = -I$(BASE)/$(TORCH_BASE)/include
TORCH_LIBBASE = -L$(BASE)/$(TORCH_BASE)/lib
OUTPUT = $(BASE)/bin
BUILD = $(BASE)/build
OBJS = $(BASE)/build/objs

prebuild:
	mkdir $(OUTPUT)
	mkdir -p $(OBJS)

train: train.o
	g++ $(CFLAGS) -o $(OUTPUT)/$@ $(OBJS)/$^ $(TORCH_LIBBASE) $(TORCH_LIBS) $(STDLIBS)

train.o: clean prebuild
	g++ $(CFLAGS) $(TORCH_API) $(TORCH_INC) -c train.cpp -o $(OBJS)/train.o

clean:
	rm -rf qwirkle $(OUTPUT) $(build)
