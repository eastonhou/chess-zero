#pragma once
#include <vector>
#include <map>
#include <ctype.h>
#include <tuple>
#include <set>
#include <mutex>
#include <numeric>

const int BASE = 100;
const int GAMEOVER_THRESHOLD = 150 * BASE;

class move_t
{
public:
	static std::string next_board(const std::string& board, const std::pair<int, int>& move)
	{
		auto r = board;
		r[move.second] = r[move.first];
		r[move.first] = ' ';
		return r;
	}

	static std::vector<std::pair<int, int>> next_steps(const std::string& board, bool red)
	{
		std::vector<std::pair<int, int>> steps;
		for (int k = 0; k < (int)board.size(); ++k)
		{
			auto chess = board[k];
			if (side(chess) == 0 || is_red(chess) != red)
				continue;
			std::vector<int> moves;
			switch (toupper(chess))
			{
			case 'R':
				moves = rider_steps(board, k);
				break;
			case 'N':
				moves = horse_steps(board, k);
				break;
			case 'B':
				moves = elephant_steps(board, k);
				break;
			case 'A':
				moves = bishop_steps(board, k);
				break;
			case 'K':
				moves = king_steps(board, k);
				break;
			case 'C':
				moves = cannon_steps(board, k);
				break;
			case 'P':
				moves = pawn_steps(board, k);
				break;
			default:
				throw "invalid chess";
				break;
			}
			for (auto move : moves)
				steps.push_back(std::pair<int, int>((int)k, move));
		}
		return steps;
	}

	static std::vector<int> rider_steps(const std::string& board, int pos)
	{
		std::vector<int> steps;
		int px, py;
		std::tie(px, py) = position2(pos);
		auto check_add = [&](int x, int y)
		{
			auto p = position1(x, y);
			if (side(board[p]) * side(board[pos]) <= 0)
				steps.push_back(p);
			return ' ' == board[p];
		};

		for (int x = px + 1; x < 9 && check_add(x, py); ++x);
		for (int x = px - 1; x >= 0 && check_add(x, py); --x);
		for (int y = py + 1; y < 10 && check_add(px, y); ++y);
		for (int y = py - 1; y >= 0 && check_add(px, y); --y);

		return steps;
	}

	static std::vector<int> horse_steps(const std::string& board, int pos)
	{
		std::vector<int> steps;
		int px, py;
		std::tie(px, py) = position2(pos);

		const int dx[] = { -2, -2, 2, 2, -1, -1, 1, 1 };
		const int dy[] = { -1, 1, -1, 1, -2, 2, -2, 2 };

		for (int k = 0; k < 8; ++k)
		{
			if (false == valid_position(px + dx[k], py + dy[k]))
				continue;
			auto bx = dx[k] / 2;
			auto by = dy[k] / 2;
			if (board[position1(px + bx, py + by)] != ' ')
				continue;
			auto p = position1(px + dx[k], py + dy[k]);
			if (side(board[p]) * side(board[pos]) <= 0)
				steps.push_back(p);
		}
		return steps;
	}

	static std::vector<int> cannon_steps(const std::string& board, int pos)
	{
		std::vector<int> steps;
		int px, py;
		std::tie(px, py) = position2(pos);

		int counter = 0;
		auto check_add = [&](int x, int y)
		{
			auto p = position1(x, y);
			switch (counter)
			{
			case 0:
				if (' ' == board[p])
					steps.push_back(p);
				else
					++counter;
				return true;
			case 1:
				if (side(board[p]) * side(board[pos]) < 0)
					steps.push_back(p);
				if(' ' != board[p])
					++counter;
				return 1 == counter;
			default:
				return false;
			}
		};
		for (int x = px + 1; x < 9 && check_add(x, py); ++x);
		counter = 0;
		for (int x = px - 1; x >= 0 && check_add(x, py); --x);
		counter = 0;
		for (int y = py + 1; y < 10 && check_add(px, y); ++y);
		counter = 0;
		for (int y = py - 1; y >= 0 && check_add(px, y); --y);
		return steps;
	}

	static std::vector<int> elephant_steps(const std::string& board, int pos)
	{
		std::vector<int> steps;
		int px, py;
		std::tie(px, py) = position2(pos);

		const int dx[] = { -2, 2, -2, 2 };
		const int dy[] = { -2, -2, 2, 2 };
		static std::set<int> valid_py = { 0, 2, 4, 5, 7, 9 };
		for (int k = 0; k < 4; ++k)
		{
			if (false == valid_position(px + dx[k], py + dy[k]))
				continue;
			auto bx = dx[k] / 2;
			auto by = dy[k] / 2;
			if (board[position1(px + bx, py + by)] != ' ')
				continue;
			if (valid_py.end() == valid_py.find(py + dy[k]))
				continue;
			auto p = position1(px + dx[k], py + dy[k]);
			if (side(board[p]) * side(board[pos]) <= 0)
				steps.push_back(p);
		}
		return steps;
	}

	static std::vector<int> bishop_steps(const std::string& board, int pos)
	{
		std::vector<int> steps;
		int px, py;
		std::tie(px, py) = position2(pos);

		const int dx[] = { -1, 1, -1, 1 };
		const int dy[] = { -1, -1, 1, 1 };
		static std::set<int> valid_py = { 0, 1, 2, 7, 8, 9 };
		for (int k = 0; k < 4; ++k)
		{
			if (false == valid_position(px + dx[k], py + dy[k]))
				continue;
			if (px + dx[k] < 3 || px + dx[k] > 5 || valid_py.end() == valid_py.find(py + dy[k]))
				continue;
			auto p = position1(px + dx[k], py + dy[k]);
			if (side(board[p]) * side(board[pos]) <= 0)
				steps.push_back(p);
		}
		return steps;
	}

	static std::vector<int> king_steps(const std::string& board, int pos)
	{
		std::vector<int> steps;
		int px, py;
		std::tie(px, py) = position2(pos);

		const int dx[] = { -1, 1, 0, 0 };
		const int dy[] = { 0, 0, -1, 1 };
		static std::set<int> valid_py = { 0, 1, 2, 7, 8, 9 };

		for (int k = 0; k < 4; ++k)
		{
			if (false == valid_position(px + dx[k], py + dy[k]))
				continue;
			if (px + dx[k] < 3 || px + dx[k] > 5 || valid_py.end() == valid_py.find(py + dy[k]))
				continue;
			auto p = position1(px + dx[k], py + dy[k]);
			if (side(board[p]) * side(board[pos]) <= 0)
				steps.push_back(p);
		}
		auto inc = py <= 2 ? 1 : -1;
		for (int y = py + inc; y >= 0 && y < 10; y += inc)
		{
			auto p = position1(px, y);
			if (' ' != board[p])
			{
				if ('K' == toupper(board[p]))
					steps.push_back(p);
				break;
			}
		}
		return steps;
	}

	static std::vector<int> pawn_steps(const std::string& board, int pos)
	{
		std::vector<int> steps;
		int px, py;
		std::tie(px, py) = position2(pos);
		const int dx[] = { 0, -1, 1 };
		const int dy[] = { -1, 0, 0 };
		int reverse;
		int count;

		auto red_king_pos = (int)board.find_first_of('K', 0);
		if (is_red(board[pos]) == (red_king_pos >= 45))
		{
			if (py <= 4)
				count = 3;
			else
				count = 1;
			reverse = 1;
		}
		else
		{
			if (py >= 5)
				count = 3;
			else
				count = 1;
			reverse = -1;
		}

		for (int k = 0; k < count; ++k)
		{
			if (false == valid_position(px + dx[k], py + dy[k] * reverse))
				continue;
			auto p = position1(px + dx[k], py + dy[k] * reverse);
			if (side(board[p]) * side(board[pos]) <= 0)
				steps.push_back(p);
		}
		return steps;
	}

	static bool is_red(char chess)
	{
		return 'A' <= chess && 'Z' >= chess;
	}


	static int position1(int x, int y)
	{
		return x + y * 9;
	}


	static std::pair<int, int> position2(int pos)
	{
		return std::pair<int, int>(pos % 9, pos / 9);
	}
	static int side(char chess)
	{
		if (is_red(chess))
			return 1;
		else if (' ' == chess)
			return 0;
		else
			return -1;
	}

	static int valid_position(int x, int y)
	{
		return x >= 0 && x < 9 && y >= 0 && y < 10;
	}
};

class rule_t
{
	friend class square_rule_t;
public:
	static std::vector<char> chess_types()
	{
		return { 'R', 'r', 'H', 'h', 'E', 'e', 'B', 'b', 'K', 'k', 'C', 'c', 'P', 'p' };
	}
	static int gameover_threshold() { return GAMEOVER_THRESHOLD; }
	static int basic_score(const std::string& board)
	{
		static std::once_flag flag;
		static std::array<int, 128> score_map = { 0 };
		std::call_once(flag, []
		{
			score_map['R'] = 10;
			score_map['H'] = 4;
			score_map['E'] = 1;
			score_map['B'] = 1;
			score_map['K'] = 300;
			score_map['C'] = 4;
			score_map['P'] = 1;
			for (int type : rule_t::chess_types())
			{
				if (islower(type))
					score_map[type] = -score_map[toupper(type)];
			}
			for (size_t k = 0; k < score_map.size(); ++k)
				score_map[k] *= BASE;
		});
		auto r = std::accumulate(board.begin(), board.end(), 0, [](int a, char b) { return a + score_map[b]; });
		return r;
	}
	static bool gameover_position(const std::string& board) {
		auto result = std::accumulate(board.begin(), board.end(), 0, [](int count, char c) {
			return count + ((c == 'K' || c == 'k') ? 1 : 0);
		});
		return result < 2;
	}
	static char flip_side(char piece)
	{
		if (move_t::is_red(piece))
			return tolower(piece);
		else if (' ' == piece)
			return piece;
		else
			return toupper(piece);
	}

	static std::string rotate_board(const std::string& board)
	{
		std::string result(board.rbegin(), board.rend());
		for(auto& ch : result) {
			ch = flip_side(ch);
		}
		return result;
	}

	static std::string initial_board() {
		std::string board = "rnbakabnr##########c#####c#p#p#p#p#p##################P#P#P#P#P#C#####C##########RNBAKABNR";
		std::replace(board.begin(), board.end(), '#', ' ');
		return board;
	}
};

class MoveTransform {
public:
	typedef std::pair<int32_t, int32_t> pos2_t;
	const static size_t action_size = 2086;
public:
	static int32_t move_to_id(const pos2_t& move) {
		static auto _m2i = _compute_m2i();
		return _m2i[move];
	}
	static const std::pair<int, int>& id_to_move(int32_t id) {
		static auto _i2m = _compute_i2m();
		return _i2m[id];
	}
	static std::vector<float> onehot(const pos2_t& move) {
		static auto _m2i = _compute_m2i();
		std::vector<float> action_probs(action_size, 0);
		action_probs[_m2i[move]] = 1;
		return action_probs;
	}
	static std::array<int, action_size> rotate_indices() {
		static auto result = _compute_rotate_indices();
		return result;
	}
	static std::array<float, action_size> map_probs(const std::vector<pos2_t>& moves, const std::vector<float>& probs) {
		static auto _m2i = _compute_m2i();
		std::array<float, action_size> result = {0};
		int i = 0;
		for (auto move : moves) {
			auto id = _m2i[move];
			result[id] = probs[i++];
		}
		return result;
	}
private:
	static std::vector<pos2_t> _compute_move_ids() {
		auto bishop_positions = _make_id_and_mirror({
			{2, 0}, {6, 0}, {0, 2}, {4, 2}, {8, 2}, {2, 4}, {6, 4}
		});
		auto adviser_positions = _make_id_and_mirror({
			{3, 0}, {5, 0}, {4, 1}, {3, 2}, {5, 2}
		});
		auto possible_moves = [&](int i0, int i1)->bool {
			auto m0 = move_t::position2(i0);
			auto m1 = move_t::position2(i1);
			if (m0 == m1) return false;
			else if (abs(m0.first - m1.first) == 1 && abs(m0.second - m1.second) == 1) {
				return adviser_positions.count(i0) && adviser_positions.count(i1);
			}
			else if (abs(m0.first - m1.first) == 2 && abs(m0.second - m1.second) == 2) {
				return bishop_positions.count(i0) && bishop_positions.count(i1);
			}
			else if (abs((m1.first - m0.first) * (m1.second - m0.second)) == 2)
				return true;
			else if ((m1.first - m0.first) * (m1.second - m0.second) == 0)
				return true;
			else
				return false;
		};
		std::set<pos2_t> moves;
		for (int i0 = 0; i0 < 90; ++i0) {
			for (int i1 = 0; i1 < 90; ++i1) {
				if (possible_moves(i0, i1))
					moves.insert({i0, i1});
			}
		}
		return std::vector<pos2_t>(moves.begin(), moves.end());
	}
	static std::map<pos2_t, int> _compute_m2i() {
		std::map<pos2_t, int> m2i;
		int id = 0;
		for (auto x : _compute_move_ids()) {
			m2i[x] = id++;
		}
		return m2i;
	}
	static std::map<int, pos2_t> _compute_i2m() {
		std::map<int, pos2_t> i2m;
		int id = 0;
		for(auto x : _compute_move_ids()) {
			i2m[id++] = x;
		}
		return i2m;
	}
	static std::array<int, action_size> _compute_rotate_indices() {
		static auto _i2m = _compute_i2m();
		static auto _m2i = _compute_m2i();
		std::array<int, action_size> result = {0};
		for (auto it = _i2m.begin(); it != _i2m.end(); ++it) {
			auto id = it->first;
			auto move = it->second;
			auto rid = _m2i[{89 - move.first, 89 - move.second}];
			result[id] = rid;
		}
		return result;
	}
	static std::set<int32_t> _make_id_and_mirror(const std::set<pos2_t>& positions) {
		std::set<int32_t> ids;
		for (auto x : positions) {
			auto id = move_t::position1(x.first, x.second);
			ids.insert(id);
			ids.insert(89 - id);
		}
		return ids;
	}
};