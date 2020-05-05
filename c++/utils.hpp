#pragma once
#include <algorithm>
#include <random>
#include <time.h>
#include <torch/torch.h>

template<class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}

inline std::vector<float> random_normal(size_t n, float mean=0, float stdev=1) {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(mean, stdev);
    generator.seed(clock());
    std::vector<float> result(n);
    for (auto& x : result) {
        x = distribution(generator);
    }
    return result;
}

template<class Container, class IndicesType>
inline void reorder(const Container& source, const IndicesType& indices, Container& target) {
    for (size_t k = 0; k < source.size(); ++k) {
        target[k] = source[indices[k]];
    }
}

template<typename T>
struct tensor_t {
    torch::Tensor operator()(const T& values, const torch::Device& device) {
        return torch::tensor(values, device);
    }
};

template<typename T, size_t N>
struct tensor_t<std::array<T, N>> {
    torch::Tensor operator()(const std::array<T, N>& values, const torch::Device& device) {
        return torch::tensor(std::vector<T>(values.begin(), values.end()), device);
    }
};

template<template<class> class Cty0, template<class> class Cty1, typename T>
struct tensor_t<Cty0<Cty1<T>>> {
    torch::Tensor operator()(const Cty0<Cty1<T>>& values, const torch::Device& device) {
        std::vector<torch::Tensor> result;
        for (auto& v : values) {
            result.push_back(tensor_t<Cty1<T>>()(v, device));
        }
        return torch::stack(result, 0);
    }
};

template<template<class> class Cty0, typename T, size_t N>
struct tensor_t<Cty0<std::array<T, N>>> {
    torch::Tensor operator()(const Cty0<std::array<T, N>>& values, const torch::Device& device) {
        std::vector<torch::Tensor> result;
        for (auto& v : values) {
            result.push_back(tensor_t<std::array<T, N>>()(v, device));
        }
        return torch::stack(result, 0);
    }
};
