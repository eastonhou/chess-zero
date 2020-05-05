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

    std::list<std::tuple<record_t, target_t>> play(int nocapture=60) {
        std::list<std::tuple<record_t, target_t>> train_data;
        int nocapture_counter = 0;
        auto board = rule_t::initial_board();
        int side = 1;
        int winner = 0;
        while (!rule_t::gameover_position(board)) {
            auto result = mcts::ponder(_model, board, side);
            auto& move = result._0;
            auto& probs = result._1;
            auto record = std::make_tuple(board, side);
            auto target = std::make_tuple(probs, 0);
            std::tuple<record_t, target_t> train_record = std::make_tuple(record, target);
            train_data.push_back(train_record);
            if (board[move[1]] != ' ') nocapture_counter = 0;
            else if (++nocapture_counter >= nocapture) break;
            board = move_t::next_board(board, move);
            side = -side;
        }
        if (board.find('K') == std::npos) winner = -1;
        else if (board.find('k') == std::npos) winner = 1;
        for (auto& x : train_data) x._1._1 = winner;
        return train_data;
    }
};