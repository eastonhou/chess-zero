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
