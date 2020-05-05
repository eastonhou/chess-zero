#pragma once
#include <torch/torch.h>
#include <torch/nn.h>
#include <list>
#include <string>
#include "definitions.hpp"
#include "rules.hpp"

class model_t : torch::nn::Module {
private:
	torch::nn::Embedding _embeddings;
	torch::nn::Sequential _input_layer;
	torch::nn::ModuleList _residual_blocks;
	torch::nn::Sequential _policy_head;
	torch::nn::Sequential _policy_projection;
	torch::nn::Sequential _value_head;
	torch::nn::Sequential _value_projection;

public:
	model_t(int num_residual_blocks=7, int embedding_dim=80)
	: _embeddings(torch::nn::Embedding(torch::nn::EmbeddingOptions(15, embedding_dim).padding_idx(0)))
	, _input_layer(_make_input_module(embedding_dim))
	, _residual_blocks(_make_residual_blocks(num_residual_blocks))
	, _policy_head(_make_policy_head())
	, _policy_projection(_make_policy_projection())
	, _value_head(_make_value_head())
	, _value_projection(_make_value_projection()) {
		register_module("embeddings", _embeddings);
		register_module("input_layer", _input_layer);
		register_module("residual_blocks", _residual_blocks);
		register_module("policy_head", _policy_head);
		register_module("policy_projection", _policy_projection);
		register_module("value_head", _value_head);
		register_module("value_projection", _value_projection);
	}
	template<template<class> class Container>
	std::tuple<torch::Tensor, torch::Tensor> forward_some(const Container<record_t>& records) {
		auto inputs = _convert_inputs(records);
		auto results = forward(inputs);
		for (size_t k = 0; k < inputs.size(0); ++k) {
			auto side = records[k].side;
			auto& p = results._0;
			auto& v = results._1;
			if (side == -1) {
				p[k] = p[k, MoveTransform::rotate_indices()];
				v[k] = -v[k];
			}
		}
		return results;
	}
	std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& inputs) {
		auto embeddings = _embeddings(inputs).permute({0, 3, 1, 2}).contiguous();
		auto x = _input_layer->forward(embeddings);
		for (auto& m : _residual_blocks) {
			auto a = m->forward(x);
			x = torch::nn::functional::relu(a + x);
		}
		auto p = _run_head(_policy_head, _policy_projection, x);
		auto v = _run_head(_value_head, _value_projection, x).view(-1);
		return std::make_tuple(p, v);
	}
	template<template<class> class Container>
	void update_policy(
		std::shared_ptr<torch::optim::Optimizer> optimizer,
		const Container<train_record_t>& records,
		size_t epochs=10) {
		std::list<record_t> inputs;
		std::list<label_t> labels;
		std::list<int> sides;
		for (auto& x : records) {
			inputs.push_back(x.input);
			labels.push_back(x.label);
			sides.push_back(x.input.side);
		}
		auto targets = _convert_targets(labels, sides);
		float tloss = 0;
		for (size_t _e = 0; _e < epochs; ++_e) {
			auto logits = forward(inputs);
			auto tp = tensor(targets._0);
			auto tv = tensor(targets._1);
			ploss = (-logits._0*tp).sum()
			vloss = torch::nn::functional::mse_loss(logits._1, tv);
			auto loss = ploss + vloss;
			optimizer.zero_grad();
			loss.backward();
			optimizer->step();
			tloss += loss.item();
		}
		std::cout << "LOSS: " << tloss/inputs.size() << std::endl;
	}
	std::shared_ptr<torch::optim::Optimizer> create_optimizer() {
		auto optimizer = std::make_shared<torch::optim::Adam>(parameters());
		return std::static_pointer_cast<torch::optim::Optimizer>(optimizer);
	}
private:
	template<template<class> class Container>
	std::tuple<std::vector<std::array<int, 90>>, std::vector<float>> _convert_targets(
		const Container<label_t>& labels, const Container<int> sides) {
		std::list<std::array<float, MoveTransform::action_size>> ps;
		std::vector<float> vs;
		for (size_t k = 0; k < labels.size(); ++k) {
			if (sides[k] == -1) {
				auto& p = labels[k]._0;
				auto& v = labels[k]._1;
				ps.push_back(p[MoveTransform::rotate_indices()]);
				vs.push_back(-v);
			}
			else {
				ps.push_back(labels[k]._0);
				vs.push_back(labels[k]._1);
			}
			return {ps, vs};
		}
	}
	template<template<class> class Container>
	torch::Tensor _convert_inputs(const Container<record_t>& records) {
		static std::map<char, int> piece_map = {
			{' ', 0}, {'r', 1}, {'n', 2}, {'b', 3}, {'a', 4}, {'k', 5}, {'c', 6}, {'p', 7},
			{'R', 8}, {'N', 9}, {'B', 10}, {'A', 11}, {'K', 12}, {'C', 13}, {'P', 14}
		}
		std::vector<std::array<int, 90>> results(boards.size());
		for (size_t k = 0; k < records.size(); ++k) {
			auto& record = records[k];
			auto side = record._1;
			auto board = side == 1 ? record._0 : rule_t::rotate_board(record._0);
			auto& data = results[k];
			for (size_t i = 0; i < 90; ++i) {
				data[i] = piece_map[board[i]];
			}
		}
		auto inputs = tensor(results).reshape({10, 9});
		return inputs;
	}
	torch::nn::Sequential _make_input_module(int embedding_dim) {
		return torch::nn::Sequential(
			_conv(embedding_dim, 128, 3),
			torch::nn::BatchNorm2d(128),
			torch::nn::ReLU(torch::nn::ReLUOptions(true))
		);
	}
	torch::nn::ModuleList _make_residual_blocks(int num_residual_blocks) {
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
		auto x = head->forward(values).reshape(values.size(0), -1);
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
			torch::nn::Linear(180, ACTION_SIZE),
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