#pragma once
#include <vector>
#include <ctype.h>
#include <tuple>
#include <set>
#include <mutex>
#include <numeric>

const int BASE = 100;
const int GAMEOVER_THRESHOLD = 150 * BASE;

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
		static int score_map[128] = { 0 };
		std::call_once(flag, []
		{
			score_map['R'] = 10;
			score_map['H'] = 4;
			score_map['E'] = 1;
			score_map['B'] = 1;
			score_map['K'] = 300;
			score_map['C'] = 4;
			score_map['P'] = 1;
			for (auto type : rule_t::chess_types())
			{
				if (islower(type))
					score_map[type] = -score_map[toupper(type)];
			}
			for (size_t k = 0; k < _countof(score_map); ++k)
				score_map[k] *= BASE;
		});
		auto r = std::accumulate(board.begin(), board.end(), 0, [](int a, char b) { return a + score_map[b]; });
		return r;
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
		return std::string(board.rbegin(), board.rend());
	}
};

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
				throw std::exception("invalid chess");
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
