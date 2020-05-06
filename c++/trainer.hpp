#include <iostream>
#include <torch/torch.h>
#include "utils.hpp"
#include "models.hpp"
#include "rules.hpp"
#include "mcts.hpp"
class Trainer {
private:
    model_t _model;
    std::shared_ptr<torch::optim::Optimizer> _optimizer;
public:
    Trainer(): _model(load_model()), _optimizer(create_optimizer(_model)) {
        auto device = torch::Device(c10::DeviceType::CUDA);
        _model.to(device);
    }
    void run() {
        std::cout << "Start training..." << std::endl;
        xtimer_t timer;
        for (auto epoch = 0;; ++epoch) {
            auto train_data = play();
            auto loss = update_policy(_model, _optimizer, train_data);
            save_model(_model);
            auto elapsed = timer.check("epoch");
            std::cout
                << "[" << epoch << "]"
                << " LOSS=" << loss
                << " STEPS=" << train_data.size()
                << " ELAPSE=" << elapsed
                << std::endl;
        }
    }
    std::list<train_record_t> play(int nocapture=60) {
        std::list<train_record_t> train_data;
        int nocapture_counter = 0;
        auto board = rule_t::initial_board();
        int side = 1;
        int winner = 0;
        while (!rule_t::gameover_position(board)) {
            action_t move;
            action_probs_t probs;
            std::tie(move, probs) = mcts_t::ponder(_model, board, side);
            record_t record = {board, side};
            label_t label = {probs, 0};
            train_record_t train_record = {record, label};
            train_data.push_back(train_record);
            if (board[move.to] != ' ') nocapture_counter = 0;
            else if (++nocapture_counter >= nocapture) break;
            print_move(train_data.size(), board, move);
            board = move_t::next_board(board, move);
            side = -side;
        }
        if (board.find('K') == std::string::npos) winner = -1;
        else if (board.find('k') == std::string::npos) winner = 1;
        for (auto& x : train_data) x.label.winner = winner;
        return train_data;
    }
    void print_move(size_t step, const std::string& board, const action_t& move) {
        std::string side = move_t::side(board[move.from]) == 1 ? "RED" : "BLACK";
        char capture = board[move.to];
        std::cout << "\r[" << step << "] " << side << ": " << "(" << move.from << "," << move.to << ")";
        if (capture != ' ')
            std::cout << " CAPTURE=" << capture;
        std::cout << "        " << std::flush;
        if (std::toupper(capture) == 'K')
            std::cout << std::endl;
    }
};