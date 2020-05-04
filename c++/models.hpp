#pragma once
#include <torch/torch.h>
#include <torch/nn.h>
#include <list>

class model_t : torch::nn::Module {
private:
	torch::nn::Sequential _input_layer;
	torch::nn::ModuleList _residual_blocks;
public:
	model_t(int num_residual_blocks=7) {
		_input_layer = make_input_module();
		_residual_blocks = make_residual_blocks(num_residual_blocks);
	}

private:
	torch::nn::Sequential make_input_module() {
		return torch::nn::Sequential(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(14, 128, 3).stride(1)),
			torch::nn::BatchNorm2d(128),
			torch::nn::ReLU()
		);
	}
	torch::nn::ModuleList make_residual_blocks(int num_residual_blocks) {
		torch::nn::ModuleList blocks;
		for (int k = 0; k < num_residual_blocks; ++k) {
			blocks->push_back(_make_residual_block());
		}
		return blocks;
	}
private:
	torch::nn::Sequential _make_residual_block() {
		return torch::nn::Sequential(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1)),
			torch::nn::BatchNorm2d(128),
			torch::nn::ReLU(),
			torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1)),
			torch::nn::BatchNorm2d(128)
		);
	}
};