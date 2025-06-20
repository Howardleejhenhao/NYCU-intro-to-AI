import numpy as np
import time
import copy

class Board:
    def __init__(self, row=6, column=7, detail=True):
        self.column = column
        self.row = row
        self.table = np.asarray([[0] * column for _ in range(row)])
        self.mark = 1           # 1 for Player 1, 2 for Player 2
        self.connect = 4        # Number of pieces needed to win
        self.cnt = 1            # Move counter
        self.round = 1
        self.valid = list(range(self.column))
        self.last = -1          # Last column played
        self.detail = detail

    def print(self, dep=0):
        for row in self.table:
            print("\t" * dep, end="")
            for point in row:
                print(point, end=" ")
            print()

    def put(self, col):
        """Place a piece in column 'col' for the current player (self.mark)."""
        if col < 0 or col >= self.column:
            return False
        if self.table[0][col] != 0:  # Column full
            return False

        # Place piece in the lowest available cell.
        for row in range(self.row - 1, -1, -1):
            if self.table[row][col] == 0:
                self.table[row][col] = self.mark
                break

        self.mark = 3 - self.mark  # Switch players.
        self.cnt += 1
        self.last = col
        self.valid = [c for c in range(self.column) if self.table[0][c] == 0]
        return True

    def win(self, piece):
        # Check horizontal win.
        for r in range(self.row):
            for c in range(self.column - (self.connect - 1)):
                window = list(self.table[r, c:c + self.connect])
                if window.count(piece) == self.connect:
                    return True
        # Check vertical win.
        for r in range(self.row - (self.connect - 1)):
            for c in range(self.column):
                window = list(self.table[r:r + self.connect, c])
                if window.count(piece) == self.connect:
                    return True
        # Check positively sloped diagonals.
        for r in range(self.row - (self.connect - 1)):
            for c in range(self.column - (self.connect - 1)):
                window = list(self.table[range(r, r + self.connect), range(c, c + self.connect)])
                if window.count(piece) == self.connect:
                    return True
        # Check negatively sloped diagonals.
        for r in range(self.connect - 1, self.row):
            for c in range(self.column - (self.connect - 1)):
                window = list(self.table[range(r, r - self.connect, -1), range(c, c + self.connect)])
                if window.count(piece) == self.connect:
                    return True
        return False

    def terminate(self):
        # The game terminates if a win is detected or if the board is full.
        if self.win(self.mark) or self.win(3 - self.mark):
            return True
        return (self.cnt == self.row * self.column)

    def start(self, agents):
        """
        Main loop for playing the game with the given two 'agents' (callable functions).
        Returns 1 if Player 1 wins, 2 if Player 2 wins, or 0 for a draw.
        """
        while not self.terminate():
            if self.detail:
                self.print()
            start_time = time.time()

            move_col = agents[self.mark - 1](self)
            if not self.put(move_col):
                print("Invalid input from agent. Column:", move_col)
                break

            end_time = time.time()
            if self.detail:
                print('Used %.3fs for this step.' % (end_time - start_time))
            self.round += 1

        print("Game finished.")
        if self.win(1):
            print("Player1 Win!!")
            print("========================================")
            return 1
        elif self.win(2):
            print("Player2 Win!!")
            print("========================================")
            return 2
        else:
            print("It's a draw game.")
            print("========================================")
            return 0

#
# Helper functions:
#
def drop_piece(board, col):
    """Return a deep copy of 'board' after dropping a piece in column 'col'."""
    nxt = copy.deepcopy(board)
    nxt.put(col)
    return nxt

def check_winning_move(board, col, piece):
    """Check if dropping a piece in column 'col' immediately produces a win."""
    next_grid = drop_piece(board, col)
    return next_grid.win(piece)

def score_move(board, col):
    """Return a heuristic score for the board after a move in column 'col'."""
    next_grid = drop_piece(board, col)
    return get_heuristic(next_grid)

def get_heuristic(board):
    """
    Evaluate the board from Player 1's perspective.
    Returns a large positive value if Player 1 is winning, a large negative value if Player 2 is winning,
    and intermediate values based on partial connect patterns.
    """
    num_twos       = count_windows(board, 2, 1)
    num_threes     = count_windows(board, 3, 1)
    num_twos_opp   = count_windows(board, 2, 2)
    num_threes_opp = count_windows(board, 3, 2)

    score = (
         1e10 * board.win(1)
       + 1e6  * num_threes
       + 10   * num_twos
       - 10   * num_twos_opp
       - 1e6  * num_threes_opp
       - 1e10 * board.win(2)
    )
    return score


def check_window(board, window, num_discs, piece):
    """Return True if 'window' contains exactly 'num_discs' of 'piece' and the rest are empty."""
    return (window.count(piece) == num_discs and window.count(0) == board.connect - num_discs)

def count_windows(board, num_discs, piece):
    """Count the number of contiguous segments ('windows') of length board.connect with 'num_discs' of 'piece'."""
    num_windows = 0
    # Horizontal windows.
    for r in range(board.row):
        for c in range(board.column - (board.connect - 1)):
            window = list(board.table[r, c:c + board.connect])
            if check_window(board, window, num_discs, piece):
                num_windows += 1
    # Vertical windows.
    for r in range(board.row - (board.connect - 1)):
        for c in range(board.column):
            window = list(board.table[r:r + board.connect, c])
            if check_window(board, window, num_discs, piece):
                num_windows += 1
    # Positive diagonal windows.
    for r in range(board.row - (board.connect - 1)):
        for c in range(board.column - (board.connect - 1)):
            window = list(board.table[range(r, r + board.connect), range(c, c + board.connect)])
            if check_window(board, window, num_discs, piece):
                num_windows += 1
    # Negative diagonal windows.
    for r in range(board.connect - 1, board.row):
        for c in range(board.column - (board.connect - 1)):
            window = list(board.table[range(r, r - board.connect, -1), range(c, c + board.connect)])
            if check_window(board, window, num_discs, piece):
                num_windows += 1
    return num_windows
