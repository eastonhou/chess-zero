#pragma once
#include <torch/torch.h>
#include <torch/nn.h>
#include <torch/script.h>
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
typedef torch::jit::script::Module model_t;

void save_model(model_t model, const std::string& path="checkpoints/model.pt") {
	auto folder = fs::path(path).parent_path();
	if (!fs::exists(folder))
		fs::create_directory(folder);
	model.save(path);
	std::cout << "Saved to checkpoint." << std::endl;
}

model_t load_model(const std::string& path="checkpoints/model.pt") {
	auto model = torch::jit::load(path);
	std::cout << "Loaded from checkpoint." << std::endl;
	return model;
}

std::shared_ptr<torch::optim::Optimizer> create_optimizer(model_t model) {
	std::vector<torch::Tensor> parameters;
	int64_t num_parameters = 0;
	for (const auto& parameter : model.parameters()) {
		parameters.push_back(parameter);
		num_parameters += parameter.numel();
	}
	auto optimizer = std::make_shared<torch::optim::Adam>(parameters);
	std::cout << "Total Parameters: " << num_parameters << std::endl;
	return std::static_pointer_cast<torch::optim::Optimizer>(optimizer);
}

torch::Device model_device(model_t model) {
	const auto& first = *model.parameters().begin();
	return first.device();
}

template<typename Ty>
torch::Tensor tensor(const Ty& values, const c10::Device& device) {
	return tensor_t<Ty>()(values, device);
}

std::tuple<torch::Tensor, torch::Tensor> _convert_outputs(torch::jit::IValue values) {
	auto tuple = values.toTuple();
	auto ps = tuple->elements()[0].toTensor();
	auto vs = tuple->elements()[1].toTensor();
	return std::make_tuple(ps, vs);
}

template<template<class> class Container>
std::tuple<torch::Tensor, torch::Tensor> forward_some(model_t model, const Container<record_t>& records) {
	auto device = model_device(model);
	auto inputs = _convert_inputs(records, device);
	auto results = _convert_outputs(model.forward(inputs));
	for (size_t k = 0; k < records.size(); ++k) {
		auto side = records[k].side;
		auto& p = std::get<0>(results);
		auto& v = std::get<1>(results);
		if (side == -1) {
			auto indices = tensor(MoveTransform::rotate_indices(), device);
			p[k] = p[k].index(indices);
			v[k] = -v[k];
		}
	}
	return results;
}

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
std::vector<torch::jit::IValue> _convert_inputs(const Container<record_t>& records, const c10::Device& device) {
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
	auto inputs = tensor(results, device).reshape({-1, 10, 9});
	std::vector<torch::jit::IValue> jit_inputs;
	jit_inputs.push_back(inputs);
	return jit_inputs;
}

template<template<class> class Container>
void update_policy(
	model_t model,
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
	auto device = model_device(model);
	auto inputs = _convert_inputs(input_records, device);
	auto targets = _convert_targets(labels, sides);
	float tloss = 0;
	for (size_t _e = 0; _e < epochs; ++_e) {
		auto logits = _convert_outputs(model.forward(inputs));
		auto tp = tensor(std::get<0>(targets), device);
		auto tv = tensor(std::get<1>(targets), device).to(torch::ScalarType::Float);
		auto probs = std::get<0>(logits);
		auto values = std::get<1>(logits);
		auto ploss = (-probs*tp).sum();
		auto vloss = torch::nn::functional::mse_loss(values, tv);
		auto loss = ploss + vloss;
		optimizer->zero_grad();
		loss.backward();
		optimizer->step();
		tloss += loss.item().toFloat();
	}
	std::cout << "LOSS: " << tloss/input_records.size()/epochs << std::endl;
}
