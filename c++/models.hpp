#pragma once
#include <torch/torch.h>
#include <torch/nn.h>
#include <list>
#include <string>
#include "rules.hpp"

class model_t : torch::nn::Module {
private:
	torch::nn::Sequential _input_layer;
	torch::nn::ModuleList _residual_blocks;

public:
	model_t(int num_residual_blocks=7, int embedding_dim=80) {
		_input_layer = make_input_module(embedding_dim);
		_residual_blocks = make_residual_blocks(num_residual_blocks);
	}

private:
	torch::nn::Sequential make_input_module(int embedding_dim) {
		return torch::nn::Sequential(
			_conv(embedding_dim, 128, 3),
			torch::nn::BatchNorm2d(128),
			torch::nn::ReLU(torch::nn::ReLUOptions(true))
		);
	}
	torch::nn::ModuleList make_residual_blocks(int num_residual_blocks) {
		torch::nn::ModuleList blocks;
		for (int k = 0; k < num_residual_blocks; ++k) {
			blocks->push_back(_make_residual_block());
		}
		return blocks;
	}
	c10::Device device() {
		return parameters()[0].device();
	}
	template<typename Ty>
	torch::Tensor tensor(const Ty& values) {
		return torch::tensor(values, device());
	}

private:
	template<typename Head, typename Projection>
	torch::Tensor _run_head(Head& head, Projection& projection, const torch::Tensor& values) {
		auto x = head(values).reshape(values.size(0), -1);
		auto y = projection(x);
		return y;
	}
	torch::nn::Sequential _make_residual_block() {
		return torch::nn::Sequential(
			_conv(128, 128, 3),
			torch::nn::BatchNorm2d(128),
			torch::nn::ReLU(torch::nn::ReLUOptions(true)),
			_conv(128, 128, 3),
			torch::nn::BatchNorm2d(128)
		);
	}
	torch::nn::Sequential _make_policy_head() {
		return torch::nn::Sequential(
			_conv(128, 2, 1),
			torch::nn::BatchNorm2d(2),
			_relu()
		);
	}
	torch::nn::Sequential _make_policy_projection() {
		return torch::nn::Sequential(
			torch::nn::Linear(180, MoveTransform::action_size),
			torch::nn::LogSoftmax(-1)
		);
	}
	torch::nn::Sequential _make_value_head() {
		return torch::nn::Sequential(
			_conv(128, 1, 1),
			torch::nn::BatchNorm2d(1),
			_relu()
		);
	}
	torch::nn::Sequential _make_value_projection() {
		return torch::nn::Sequential(
			torch::nn::Linear(90, 256),
			_relu(),
			torch::nn::Linear(256, 1),
			torch::nn::Tanh()
		);
	}
	torch::nn::Conv2d _conv(int in_channels, int out_channels, int kernel_size) {
		auto options = torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
			.stride(1).padding(kernel_size / 2);
		return torch::nn::Conv2d(options);
	}
	torch::nn::ReLU _relu() {
		return torch::nn::ReLU(torch::nn::ReLUOptions(true));
	}
};

template<>
torch::Tensor model_t::tensor<torch::Tensor>(const torch::Tensor& values) {
	return values.to(c10::TensorOptions(device()));
}