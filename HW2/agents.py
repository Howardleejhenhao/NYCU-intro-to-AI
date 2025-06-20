import numpy as np
import random
import game

def print_INFO():
    """
    Prints your homework submission details.
    Please replace the placeholders (date, name, student ID) with valid information
    before submitting.
    """
    print(
        """========================================
        DATE: 2025/03/25
        STUDENT NAME: 李振皞
        STUDENT ID: 113550046
        ========================================
        """)


#
# Basic search functions: Minimax and Alpha‑Beta
#

def minimax(grid, depth, maximizingPlayer, dep=4):
    """
    TODO (Part 1): Implement recursive Minimax search for Connect Four.

    Return:
      (boardValue, {setOfCandidateMoves})

    Where:
      - boardValue is the evaluated utility of the board state
      - {setOfCandidateMoves} is a set of columns that achieve this boardValue
    """
    if grid.terminate() or depth == 0:
      return get_heuristic(grid), set()
    could_move = [col for col in range(len(grid.table[0])) if grid.table[0][col] == 0]
    if len(could_move) == 0:
      return get_heuristic(grid), set()
    inf = 1e9
    if maximizingPlayer == 1:
      best = -inf
      nxt_moves = set()
      for col in could_move:
        nw_grid = game.drop_piece(grid, col)
        value, st = minimax(nw_grid, depth - 1 , False, dep)
        if value > best:
          best = value
          nxt_moves.clear();
          nxt_moves.add(col)
        elif value == best:
          nxt_moves.add(col)
      return best, nxt_moves
    else:
      best = inf
      nxt_moves = set()
      for col in could_move:
        nw_grid = game.drop_piece(grid, col)
        value, st = minimax(nw_grid, depth - 1, True, dep)
        if value < best:
          best = value
          nxt_moves.clear()
          nxt_moves.add(col)
        elif value == best:
          nxt_moves.add(col)
      return best, nxt_moves

def alphabeta(grid, depth, maximizingPlayer, alpha, beta, dep=4):
    """
    TODO (Part 2): Implement Alpha-Beta pruning as an optimization to Minimax.

    Return:
      (boardValue, {setOfCandidateMoves})

    Where:
      - boardValue is the evaluated utility of the board state
      - {setOfCandidateMoves} is a set of columns that achieve this boardValue
      - Prune branches when alpha >= beta
    """
    # Placeholder return to keep function structure intact
    if depth == 0 or grid.terminate():
      return get_heuristic(grid), set()
    could_moves = [col for col in range(len(grid.table[0])) if grid.table[0][col] == 0]
    inf = 1e100
    if len(could_moves) == 0:
      return get_heuristic(grid), set()
    if maximizingPlayer:
      best = -inf
      nxt_move = set()
      for col in could_moves:
        nw_grid = game.drop_piece(grid, col)
        value, st = alphabeta(nw_grid, depth - 1, False, alpha, beta, dep)
        if value > best:
          best = value
          nxt_move = {col}
        elif value == best:
          nxt_move.add(col)

        alpha = max(alpha, best)
        if alpha >= beta:
          break
      return best, nxt_move
    else:
      best = inf
      nxt_move = set()
      for col in could_moves:
        nw_grid = game.drop_piece(grid, col)
        value, st = alphabeta(nw_grid, depth - 1, True, alpha, beta, dep)
        if value < best:
          best = value
          nxt_move = {col}
        elif value == best:
          nxt_move.add(col)
        beta = min(beta, best)
        if alpha >= beta:
          break
    return best, nxt_move


#
# Basic agents
#

def agent_minimax(grid):
    """
    Agent that uses the minimax() function with a default search depth (e.g., 4).
    Must return a single column (integer) where the piece is dropped.
    """
    return random.choice(list(minimax(grid, 4, True)[1]))


def agent_alphabeta(grid):
    """
    Agent that uses the alphabeta() function with a default search depth (e.g., 4).
    Must return a single column (integer) where the piece is dropped.
    """
    return random.choice(list(alphabeta(grid, 4, True, -np.inf, np.inf)[1]))


def agent_reflex(grid):
    """
    A simple reflex agent provided as a baseline:
      - Checks if there's an immediate winning move.
      - Otherwise picks a random valid column.
    """
    wins = [c for c in grid.valid if game.check_winning_move(grid, c, grid.mark)]
    if wins:
        return random.choice(wins)
    return random.choice(grid.valid)


def agent_strong(grid):
    """
    TODO (Part 3): Design your own agent (depth = 4) to consistently beat the Alpha-Beta agent (depth = 4).
    This agent will typically act as Player 2.
    """
    # Placeholder logic that calls your_function().
    return random.choice(list(your_function(grid, 4, False, -np.inf, np.inf)[1]))


#
# Heuristic functions
#

def get_heuristic(board):
    """
    Evaluates the board from Player 1's perspective using a basic heuristic.

    Returns:
      - Large positive value if Player 1 is winning
      - Large negative value if Player 2 is winning
      - Intermediate scores based on partial connect patterns
    """
    num_twos       = game.count_windows(board, 2, 1)
    num_threes     = game.count_windows(board, 3, 1)
    num_twos_opp   = game.count_windows(board, 2, 2)
    num_threes_opp = game.count_windows(board, 3, 2)

    score = (
          1e10 * board.win(1)
        + 1e6  * num_threes
        + 10   * num_twos
        - 10   * num_twos_opp
        - 1e6  * num_threes_opp
        - 1e10 * board.win(2)
    )
    return score


def get_heuristic_strong(board):
    """
    TODO (Part 3): Implement a more advanced board evaluation for agent_strong.
    Currently a placeholder that returns 0.
    """
    num_twos       = game.count_windows(board, 2, 1)
    num_threes     = game.count_windows(board, 3, 1)
    num_twos_opp   = game.count_windows(board, 2, 2)
    num_threes_opp = game.count_windows(board, 3, 2)
    total_cnt = board.cnt
    score = (
          1e10 * board.win(1)
        + 1e6 * total_cnt * num_threes
        + 1000 * total_cnt * num_twos
        - 1000 * total_cnt * num_twos_opp
        - 1e6  * total_cnt * num_threes_opp
        - 1e10 * board.win(2)
    )


    width = board.table.shape[1]
    center_bonus = 0
    for col in range(width):
        if col == 0 or col == width - 1:
            weight = 4e4 if total_cnt <= 25 else 1e3
        elif col == 1 or col == width - 2:
            weight = 6e4 if total_cnt <= 25 else 2e3
        elif col == 2 or col == width - 3:
            weight = 8e4 if total_cnt <= 25 else 3e3
        else:
            weight = 1e5 if total_cnt <= 25 else 4e3
        col_pieces = board.table[:, col]
        center_bonus += col_pieces.tolist().count(1) * weight
        center_bonus -= col_pieces.tolist().count(2) * weight
        if board.table[0][col] == 0:
          score += weight // 10
    score += center_bonus

    return score


def your_function(grid, depth, maximizingPlayer, alpha, beta, dep=4):
    """
    A stronger search function that uses get_heuristic_strong() instead of get_heuristic().
    You can employ advanced features (e.g., improved move ordering, deeper lookahead).

    Return:
      (boardValue, {setOfCandidateMoves})

    Currently a placeholder returning (0, {0}).
    """
    if depth == 0 or grid.terminate():
      return get_heuristic_strong(grid), set()
    could_moves = [col for col in range(len(grid.table[0])) if grid.table[0][col] == 0]
    inf = 1e100
    if len(could_moves) == 0:
      return get_heuristic_strong(grid), set()
    if maximizingPlayer:
      best = -inf
      nxt_move = set()
      for col in could_moves:
        nw_grid = game.drop_piece(grid, col)
        value, st = your_function(nw_grid, depth - 1, False, alpha, beta, dep)
        if value > best:
          best = value
          nxt_move = {col}
        elif value == best:
          nxt_move.add(col)

        alpha = max(alpha, best)
        if alpha >= beta:
          break
      return best, nxt_move
    else:
      best = inf
      nxt_move = set()
      for col in could_moves:
        nw_grid = game.drop_piece(grid, col)
        value, st = your_function(nw_grid, depth - 1, True, alpha, beta, dep)
        if value < best:
          best = value
          nxt_move = {col}
        elif value == best:
          nxt_move.add(col)
        beta = min(beta, best)
        if alpha >= beta:
          break
    return best, nxt_move
