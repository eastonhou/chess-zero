#include <memory>
#include <list>
#include <set>
#include <math.h>
#include <algorithm>
#include "definitions.hpp"
#include "rules.hpp"
#include "models.hpp"
#include "utils.hpp"

typedef std::shared_ptr<class node_t> nodeptr;
class node_t : public std::enable_shared_from_this<node_t> {
public:
    std::string board;
    int side;
    nodeptr parent;
    std::vector<nodeptr> children;
    std::vector<action_t> moves;
    float P = 0;
    float W = 0;
    int N = 0;
    bool next_to_terminal = false;
    bool terminal = false;
public:
    node_t(const std::string board, int side, nodeptr parent=nullptr)
    : board(board), side(side), parent(parent) {
    }
    float Q()const {
        if (terminal) return 100000;
        else if (next_to_terminal) return -100000;
        else return N > 0 ? (W / N * -side) : 0;
    }
    float U(float c=0.5)const {
        return std::max(c*sqrt((float)parent->N)/(1+N), P);
    }
    nodeptr select() {
        float best = -1E18;
        nodeptr node = nullptr;
        for (auto& child : children) {
            auto score = child->Q() + child->U();
            if (score > best) {
                best = score;
                node = child;
            }
        }
        assert(node != nullptr);
        return node;
    }
    nodeptr select_to_leaf() {
        auto node = this->shared_from_this();
        while (node->children.size() > 0) {
            node = node->select();
        }
        return node;
    }
    std::set<nodeptr> select_multiple(int n) {
        std::set<nodeptr> nodes;
        for (int k = 0; k < n; ++k) {
            auto node = select_to_leaf();
            if (nodes.count(node)) break;
            node->backup(node->side*1000);
            nodes.insert(node);
            if (node->terminal) break;
        }
        for (auto& node : nodes) node->backup(node->side*1000, -1);
        return nodes;
    }
    template<template<class> class Container>
    void expand(const Container<action_t>& moves, const action_probs_t& probs) {
        this->moves = moves;
        float total_P = 0;
        for (auto& move : moves) {
            auto board = move_t::next_board(this->board, move);
            auto child = std::make_shared<node_t>(board, -this->side, this->shared_from_this());
            child->P = probs[MoveTransform::move_to_id(move)];
            total_P += child->P;
            this->children.push_back(child);
        }
        for (auto& child : this->children) {
            child->P /= total_P;
        }
    }
    void backup(float value, int direction=1) {
        auto node = this->shared_from_this();
        while (node) {
            node->W += value*direction;
            node->N += direction;
            node = node->parent;
        }
    }
    void complete() {
        parent->next_to_terminal = true;
        terminal = true;
    }
};

class state_t {
public:
    nodeptr root;
    bool terminal = false;
    state_t(const std::string& board, int side): root(std::make_shared<node_t>(board, side)) {
    }
    std::vector<float> statistics()const {
        std::vector<float> visits;
        float sum = 0;
        for (auto& x : root->children) {
            auto N = x->terminal ? 1E8 : x->N;
            visits.push_back(N);
            sum += N;
        }
        for (auto& x : visits) x /= sum;
        return visits;
    }
    void complete() {
        terminal = true;
    }
};

class mcts_t {
public:
    static void play_multiple(model_t model, state_t& state, int n) {
        auto nodes = state.root->select_multiple(n);
        std::vector<nodeptr> nonterminals;
        for (auto& node : nodes) {
            if (rule_t::gameover_position(node->board)) {
                node->complete();
                node->backup(-node->side);
            }
            else {
                nonterminals.push_back(node);
            }
        }
        if (nonterminals.empty()) return;
        std::vector<record_t> records;
        for (auto& x : nonterminals) {
            records.push_back({x->board, x->side});
        }
        auto result = model->forward_some(records);
        auto tprobs = std::get<0>(result).exp().cpu();
        auto tvalues = std::get<1>(result).cpu();
        auto probs = tprobs.contiguous().data_ptr<float>();
        auto values = tvalues.contiguous().data_ptr<float>();
        for (size_t k = 0; k < nonterminals.size(); ++k) {
            auto& node = nonterminals[k];
            auto moves = move_t::next_steps(node->board, node->side == 1);
            action_probs_t _probs;
            std::copy(&probs[k*ACTION_SIZE], &probs[(k+1)*ACTION_SIZE], _probs.begin());
            node->expand(moves, _probs);
            node->backup(values[k]);
        }
    }
    template<template<class> class Cty0, template<class> class Cty1>
    static action_t select(const Cty0<action_t>& moves, const Cty1<float>& probs, float keep) {
        if (keep >= 1) {
            auto index = argmax(probs.begin(), probs.end());
            return moves[index];
        }
        else {
            std::vector<float> probs_with_noise(probs.size());
            std::vector<float> noise = random_normal(probs.size(), 0, 0.3);
            for (size_t k = 0; k < probs.size(); ++k) {
                probs_with_noise[k] = probs[k]*keep + noise[k]*(1-keep);
            }
            auto index = argmax(probs_with_noise.begin(), probs_with_noise.end());
            return moves[index];
        }
    }
    static std::tuple<action_t, action_probs_t> ponder(
        model_t model, const std::string& board, int side, size_t playouts=200, float keep=0.75) {
        state_t state(board, side);
        for (size_t k = 0; k < playouts; ++k) {
            play_multiple(model, state, 64);
            if (state.terminal) break;
        }
        auto probs = state.statistics();
        auto move = select(state.root->moves, probs, keep);
        auto action_probs = MoveTransform::map_probs(state.root->moves, probs);
        return std::make_tuple(move, action_probs);
    }
};