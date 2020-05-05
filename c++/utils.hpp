#pragma once
#include <algorithm>
#include <random>
#include <time.h>

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