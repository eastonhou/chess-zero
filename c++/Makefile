TORCH_API = -I./libtorch/include/torch/csrc/api/include/
TORCH_LIB = -L./libtorch/lib

train : train.o
	g++ -Wall -Werror -std=c++14 -g -O -o $@ $^ $(TORCH_LIB)

train.o:
	g++ -Wall -Werror -std=c++14 -g -O $(TORCH_API) -c train.cpp

clean:
	rm -rf qwirkle *.o *.dSYM
