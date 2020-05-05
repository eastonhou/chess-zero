#include <torch/torch.h>
#include "models.hpp"
#include "rules.hpp"

class Trainer {
private:
    model_t _model;
    std::shared_ptr<torch::optim::Optimizer> _optimizer;
public:
    Trainer(): _model(), _optimizer(_model.create_optimizer()) {
    }

    void run() {
        while (true) {
            auto train_data = play();
            _model.update_policy(_optimizer, train_data);
        }
    }

    std::list<train_record_t> play(int nocapture=60) {
        std::list<train_record_t> train_data;
        int nocapture_counter = 0;
        auto board = rule_t::initial_board();
        int side = 1;
        int winner = 0;
        while (!rule_t::gameover_position(board)) {
            auto result = mcts::ponder(_model, board, side);
            auto& move = result._0;
            auto& probs = result._1;
            record_t record = {board, side};
            label_t label = {probs, 0};
            train_record_t train_record = {record, label};
            train_data.push_back(train_record);
            if (board[move[1]] != ' ') nocapture_counter = 0;
            else if (++nocapture_counter >= nocapture) break;
            board = move_t::next_board(board, move);
            side = -side;
        }
        if (board.find('K') == std::string::npos) winner = -1;
        else if (board.find('k') == std::string::npos) winner = 1;
        for (auto& x : train_data) x.label.winner = winner;
        return train_data;
    }
};