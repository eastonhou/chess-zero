#pragma once

#include <string>
#include <array>

#define ACTION_SIZE 2086

struct record_t {
    std::string board;
    int side;
};
typedef std::array<float, ACTION_SIZE> action_probs_t;
struct label_t {
    action_probs_t action_probs;
    int winner;
};

struct output_t {
    action_probs_t action_probs;
    float win_prob;
};
struct action_t {
    int from;
    int to;
    bool operator < (const action_t& other)const {
        if (from < other.from) return true;
        if (from > other.from) return false;
        return to < other.to;
    }
};

struct train_record_t {
    record_t input;
    label_t label;
};
