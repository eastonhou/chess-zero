#include "models.hpp"
#include "rules.hpp"
#include <torch/torch.h>

class Trainer {
private:
    model_t _model;
    std::shared_ptr<torch::optim::Optimizer> _optimizer;
public:
    typedef model_t::record_t record_t;
    typedef model_t::target_t target_t;
    Trainer(): _model(), _optimizer(_model.create_optimizer()) {
    }

    void run() {
        while (true) {
            auto train_data = play();
            _model.update_policy(_optimizer, train_data);
        }
    }

    std::list<std::tuple<record_t, target_t>> play() {
        std::list<std::tuple<record_t, target_t>> train_data;
        std::list<char> captuers;
        auto board = rule_t::initial_board();
        int side = 1;
        while (!rule_t::gameover_position(board)) {
            
        }
    }
};