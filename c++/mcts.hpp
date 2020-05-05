#include <memory>
#include <list>
#include <set>
#include <math.h>
#include <algorithm>
#include "definitions.hpp"
#include "rules.hpp"

typedef std::shared_ptr<class node_t> nodeptr;
class node_t : std::enable_shared_from_this<node_t> {
public:
    std::string board;
    int side;
    nodeptr parent;
    std::list<nodeptr> children;
    std::list<action_t> moves;
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
        return std::max(c*sqrt(parent->N)/(1+N), P);
    }
    nodeptr select() {
        float best = -100000;
        nodeptr node = nullptr;
        for (auto& child : children) {
            auto score = child->Q() + child->U();
            if (score > best) {
                best = score;
                node = child;
            }
        }
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
        for (auto& node : nodes) node->backup(node.side*1000, -1);
        return nodes;
    }
    template<template<class> class Container>
    void expand(const Container<action_t>& moves, const action_probs_t& probs) {
        this->moves = moves;
        float total_P = 0;
        for (auto& move : moves) {
            auto board = rule_t::next_board(this->board, move);
            auto child = std::make_ptr<node_t>(board, -this->side, this->shared_from_this());
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
        float sum;
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
