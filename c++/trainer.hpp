#include <iostream>
#include <torch/torch.h>
#include <future>
#include <list>
#include <thread>
#include "utils.hpp"
#include "models.hpp"
#include "rules.hpp"
#include "mcts.hpp"
class Trainer {
private:
    model_t _model;
    std::shared_ptr<torch::optim::Optimizer> _optimizer;
    int _epoch = 0;
public:
    Trainer(): _model(load_model()), _optimizer(create_optimizer(_model)) {
        auto device = torch::Device(c10::DeviceType::CUDA);
        _model.to(device);
    }
    void run() {
        std::list<std::thread> workers;
        xtimer_t timer;
        async_queue_t<train_record_t> queue([&](std::list<train_record_t>& queue){
            this->update(queue, timer);
        });
        for (auto k = 0; k < 8; ++k) {
            auto worker = [k, &queue, this]{
                std::cout << "Trainer " << k << " started." << std::endl;
                while (true)
                    queue.put(play());
            };
            workers.push_back(std::thread(worker));
        }
        for(auto& worker : workers) worker.join();
    }
    void update(std::list<train_record_t>& queue, xtimer_t& timer) {
        const size_t batch_size = 128;
        while (queue.size() >= batch_size) {
            std::list<train_record_t> batch;
            for (size_t k = 0; k < batch_size; ++k) {
                batch.push_back(queue.front());
                queue.pop_front();
            }
            auto loss = update_policy(_model, _optimizer, batch);
            save_model(_model);
            auto elapsed = timer.check("epoch");
            std::cout
                << "[" << _epoch++ << "]"
                << " LOSS=" << loss
                << " STEPS=" << batch.size()
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
            //print_move(board, move);
            board = move_t::next_board(board, move);
            side = -side;
        }
        if (board.find('K') == std::string::npos) winner = -1;
        else if (board.find('k') == std::string::npos) winner = 1;
        for (auto& x : train_data) x.label.winner = winner;
        return train_data;
    }
    void print_move(const std::string& board, const action_t& move) {
        std::string side = move_t::side(board[move.from]) == 1 ? "RED" : "BLACK";
        char capture = board[move.to];
        std::cout << side << " MOVE: " << "(" << move.from << "," << move.to << ")";
        if (capture != ' ')
            std::cout << " CAPTURE=" << capture;
        std::cout << std::endl;
    }
};