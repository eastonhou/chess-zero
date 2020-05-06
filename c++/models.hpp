#pragma once
#include <torch/torch.h>
#include <torch/nn.h>
#include <list>
#include <vector>
#include <array>
#include <string>
#include <iostream>
#include <experimental/filesystem>
#include "definitions.hpp"
#include "rules.hpp"
#include "utils.hpp"
namespace fs = std::experimental::filesystem::v1;
class model_imply_t : public torch::nn::Module {
private:
	torch::nn::Embedding _embeddings;
	torch::nn::Sequential _input_layer;
	torch::nn::ModuleList _residual_blocks;
	torch::nn::Sequential _policy_head;
	torch::nn::Sequential _policy_projection;
	torch::nn::Sequential _value_head;
	torch::nn::Sequential _value_projection;

public:
	model_imply_t(int num_residual_blocks=7, int embedding_dim=80)
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
		for (int k = 0; k < inputs.size(0); ++k) {
			auto side = records[k].side;
			auto& p = std::get<0>(results);
			auto& v = std::get<1>(results);
			if (side == -1) {
				auto indices = tensor(MoveTransform::rotate_indices());
				p[k] = p[k].index(indices);
				v[k] = -v[k];
			}
		}
		return results;
	}
	std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& inputs) {
		auto embeddings = _embeddings(inputs).permute({0, 3, 1, 2}).contiguous();
		auto x = _input_layer->forward(embeddings);
		for (auto& m : *_residual_blocks) {
			auto a = m->as<torch::nn::Sequential>()->forward(x);
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
		std::vector<record_t> input_records;
		std::vector<label_t> labels;
		std::vector<int> sides;
		for (auto& x : records) {
			input_records.push_back(x.input);
			labels.push_back(x.label);
			sides.push_back(x.input.side);
		}
		auto inputs = _convert_inputs(input_records);
		auto targets = _convert_targets(labels, sides);
		float tloss = 0;
		for (size_t _e = 0; _e < epochs; ++_e) {
			auto logits = forward(inputs);
			auto tp = tensor(std::get<0>(targets));
			auto tv = tensor(std::get<1>(targets)).to(torch::ScalarType::Float);
			auto ploss = (-std::get<0>(logits)*tp).sum();
			auto vloss = torch::nn::functional::mse_loss(std::get<1>(logits), tv);
			auto loss = ploss + vloss;
			optimizer->zero_grad();
			loss.backward();
			optimizer->step();
			tloss += loss.item().toFloat();
		}
		std::cout << "LOSS: " << tloss/inputs.size(0) << std::endl;
	}
	std::shared_ptr<torch::optim::Optimizer> create_optimizer() {
		auto optimizer = std::make_shared<torch::optim::Adam>(parameters());
		return std::static_pointer_cast<torch::optim::Optimizer>(optimizer);
	}
private:
	template<template<class> class Container>
	std::tuple<std::vector<std::vector<float>>, std::vector<float>> _convert_targets(
		const Container<label_t>& labels, const Container<int> sides) {
		std::vector<std::vector<float>> ps;
		std::vector<float> vs;
		for (size_t k = 0; k < labels.size(); ++k) {
			if (sides[k] == -1) {
				auto& p = labels[k].action_probs;
				auto& v = labels[k].winner;
				action_probs_t reversed_p;
				reorder(p, MoveTransform::rotate_indices(), reversed_p);
				ps.push_back(std::vector<float>(reversed_p.begin(), reversed_p.end()));
				vs.push_back(-v);
			}
			else {
				auto& p = labels[k].action_probs;
				ps.push_back(std::vector<float>(p.begin(), p.end()));
				vs.push_back(labels[k].winner);
			}
		}
		return std::make_tuple(ps, vs);
	}
	template<template<class> class Container>
	torch::Tensor _convert_inputs(const Container<record_t>& records) {
		static std::map<char, int> piece_map = {
			{' ', 0}, {'r', 1}, {'n', 2}, {'b', 3}, {'a', 4}, {'k', 5}, {'c', 6}, {'p', 7},
			{'R', 8}, {'N', 9}, {'B', 10}, {'A', 11}, {'K', 12}, {'C', 13}, {'P', 14}
		};
		std::vector<std::array<int, 90>> results(records.size());
		size_t k = 0;
		for (auto& record : records) {
			auto side = record.side;
			auto board = side == 1 ? record.board : rule_t::rotate_board(record.board);
			auto& data = results[k];
			for (size_t i = 0; i < 90; ++i) {
				data[i] = piece_map[board[i]];
			}
			++k;
		}
		auto inputs = tensor(results).reshape({-1, 10, 9});
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
		return tensor_t<Ty>()(values, device());
	}

private:
	template<typename Head, typename Projection>
	torch::Tensor _run_head(Head& head, Projection& projection, const torch::Tensor& values) {
		auto x = head->forward(values).reshape({values.size(0), -1});
		auto y = projection->forward(x);
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

TORCH_MODULE_IMPL(model_t, model_imply_t);

void save_model(model_t model, const std::string& path="checkpoints/model.pt") {
	auto folder = fs::path(path).parent_path();
	if (!fs::exists(folder))
		fs::create_directory(folder);
	torch::save(model, path);
}

void try_load_model(model_t model, const std::string& path="checkpoints/model.pt") {
	if (fs::exists(path)) {
		torch::load(model, path);
		std::cout << "loaded from checkpoint." << std::endl;
	}
}
