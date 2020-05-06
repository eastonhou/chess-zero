#pragma once
#include <algorithm>
#include <random>
#include <map>
#include <ctime>
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
        std::vector<T> data(values.size() * N);
        auto it = data.begin();
        for (auto& v : values) {
            std::copy(v.begin(), v.end(), it);
            it += v.size();
        }
        return torch::tensor(data, device).reshape({(long)values.size(), N});
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


template<typename T>
class async_queue_t {
    std::condition_variable cond;
    std::mutex mutex;
    std::queue<T> cpq;
    size_t maxsize;
public:
    async_queue_t(size_t mxsz) : maxsize(mxsz) { }

    void add(T request)
    {
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [this]()
        { return !full(); });
        cpq.push(request);
        lock.unlock();
        cond.notify_all();
    }

    T consume()
    {
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [this]()
        { return !empty(); });
        auto request = cpq.front();
        cpq.pop();
        lock.unlock();
        cond.notify_all();
        return request;
    }
    bool full() const {
        return cpq.size() >= maxsize;
    }
    bool empty() const {
        return cpq.size() == 0;
    }
    int length() const {
        return cpq.size();
    }
    void clear() {
        std::unique_lock<std::mutex> lock(mutex);
        while (!empty())
        {
            cpq.pop();
        }
        lock.unlock();
        cond.notify_all();
    }
};