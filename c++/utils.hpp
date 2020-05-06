#pragma once
#include <algorithm>
#include <random>
#include <map>
#include <ctime>
#include <functional>
#include <mutex>
#include <torch/torch.h>

template<class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}

inline std::vector<float> random_normal(size_t n, float mean=0, float stdev=1) {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(mean, stdev);
    generator.seed(std::clock());
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

class xtimer_t {
    clock_t _time = std::clock();
    static std::map<std::string, double>& _timers() {
        static std::map<std::string, double> __timers;
        return __timers;
    }
public:
    double check(const std::string& name=std::string()) {
        auto elapsed = std::clock() - _time;
        auto secs = elapsed / (double)CLOCKS_PER_SEC;
        _timers()[name] += secs;
        _time += elapsed;
        return secs;
    }
    static void print() {
        for (auto kv : _timers()) {
            std::cout << kv.first << ": " << kv.second << std::endl;
        }
    }
};

template<typename Ty>
class async_queue_t {
private:
    std::list<Ty> _queue;
    std::mutex _mutex;
    std::function<void(decltype(_queue)&)> _callback;
public:
    template<typename callback_t>
    async_queue_t(callback_t callback):_callback(callback) {}
    template<template<class> class Cty>
    void put(const Cty<Ty>& records) {
        std::unique_lock<std::mutex> lock(_mutex);
        _queue.insert(_queue.end(), records.begin(), records.end());
        _callback(_queue);
    }
};
