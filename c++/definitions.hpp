#include <string>
#include <array>

#define ACTION_SIZE 2086

struct record_t {
    std::string board;
    int side;
};

struct label_t {
    std::array<float, ACTION_SIZE> action_probs;
    int winner;
};

struct action_t {
    int from;
    int to;
};

struct train_record_t {
    record_t input;
    label_t label;
};
