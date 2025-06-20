#!/usr/bin/env python3
import pygame
import sys
import time
import numpy as np
import argparse

# Import game engine and agent functions.
from game import Board
from agents import agent_minimax, agent_alphabeta, agent_reflex, agent_strong, print_INFO

# --------------------------------------------------------------------------------------
# Pygame-based visualization constants
# --------------------------------------------------------------------------------------
SQUARESIZE = 100          # Pixel size for each board cell
RADIUS = int(SQUARESIZE / 2 - 5)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
FPS = 60                  # Frames per second

# --------------------------------------------------------------------------------------
# Utility: Draw the board state onto the Pygame window
# --------------------------------------------------------------------------------------
def draw_board(screen, board):
    """
    Draw the Connect Four board onto the Pygame surface.
    
    :param screen: The pygame display surface.
    :param board:  The Board object (from game.py).
    """
    rows = board.row
    cols = board.column
    #print("draw_board")
    # Draw the board grid and empty slots.
    for c in range(cols):
        for r in range(rows):
            pygame.draw.rect(screen, BLUE,
                             (c * SQUARESIZE, (r + 1) * SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK,
                               (int(c * SQUARESIZE + SQUARESIZE / 2),
                                int((r + 1) * SQUARESIZE + SQUARESIZE / 2)),
                               RADIUS)

    # Draw the pieces.
    for c in range(cols):
        for r in range(rows):
            piece = board.table[r][c]
            if piece == 1:
                pygame.draw.circle(screen, RED,
                                   (int(c * SQUARESIZE + SQUARESIZE / 2),
                                    int((r + 1) * SQUARESIZE + SQUARESIZE / 2)),
                                   RADIUS)
            elif piece == 2:
                pygame.draw.circle(screen, YELLOW,
                                   (int(c * SQUARESIZE + SQUARESIZE / 2),
                                    int((r + 1) * SQUARESIZE + SQUARESIZE / 2)),
                                   RADIUS)
    pygame.display.update()

# --------------------------------------------------------------------------------------
# Utility: Get human input via mouse click
# --------------------------------------------------------------------------------------
def get_human_move(event, board):
    """
    Determine the selected column based on a mouse click event.
    
    :param event: Pygame mouse event.
    :param board: The Board object.
    :return: The column index chosen by the player.
    """
    x_pos = event.pos[0]
    col = int(x_pos // SQUARESIZE)
    return col

# --------------------------------------------------------------------------------------
# GUI Mode: Interactive play (human vs. powerful agent OR student agent vs. powerful agent)
# --------------------------------------------------------------------------------------
def RunGUI(agent_name=None, agent_name2=None):
    """
    Run the Connect Four game in GUI mode.
    
    If an agent is specified via `agent_name` (using the -p flag), that agent will
    control Player 1 (the student's designed agent) while Player 2 is controlled by
    a very powerful agent. If no agent is specified, Player 1 is human-controlled.
    
    :param agent_name: (Optional) Name of the student's designed agent.
    """
    pygame.init()
    clock = pygame.time.Clock()

    # Create the game board.
    board = Board(row=6, column=7, detail=False)
    width = board.column * SQUARESIZE
    height = (board.row + 1) * SQUARESIZE  # Extra space at the top.
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Connect Four")

    # Agent mapping for potential student agents.
    agent_mapping = {
        "ReflexAgent": agent_reflex,
        "Minimax": agent_minimax,
        "AlphaBeta": agent_alphabeta,
        "StrongAgent": agent_strong
    }

    if agent_name:
        if agent_name in agent_mapping:
            player1 = agent_mapping[agent_name]
        else:
            print(f"Agent '{agent_name}' not found. Exiting.")
            pygame.quit()
            sys.exit()
    else:
        player1 = None  # Human-controlled.

 
    if agent_name2:
        if agent_name2 in agent_mapping:
            player2 = agent_mapping[agent_name2]
        else:
            print(f"Agent '{agent_name2}' not found. Exiting.")
            pygame.quit()
            sys.exit()
    else:
        player2 = agent_reflex  #Default agent is reflexAgent
    agents = [player1, player2]

    screen.fill(BLACK)
    draw_board(screen, board)

    running = True
    while running:
        # Check for terminal game state.
        if board.terminate():
            time.sleep(2)
            running = False
            break

        # Process Pygame events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # Handle human input if it's the human's turn.
            if agents[board.mark - 1] is None and event.type == pygame.MOUSEBUTTONDOWN:
                col = get_human_move(event, board)
                if not board.put(col):
                    print(f"Invalid move at column {col}.")
                else:
                    draw_board(screen, board)
                break

        # If it's an AI turn (or both players are AI).
        if agents[board.mark - 1] is not None:
            agent_func = agents[board.mark - 1]
            col = agent_func(board)
            if not board.put(col):
                print(f"Agent provided invalid move: {col}. Exiting.")
                running = False
                break
            draw_board(screen, board)
            time.sleep(0.5)  # Delay for visualization.

        clock.tick(FPS)

    # Game over: determine and display the winner.
    # If human is playing (i.e. player1 is None), display custom win/lose messages.
    if board.win(1):
        if agents[0] is None:
            winner = "Congratulations, you beat the AI agent!"
        else:
            winner = "Player 1 wins!"
    elif board.win(2):
        if agents[0] is None:
            winner = "The AI agent wins! Better luck next time."
        else:
            winner = "Player 2 wins!"
    else:
        winner = "It's a draw."
    print("Game Over! Winner:", winner)
    print_INFO()
    time.sleep(2)
    pygame.quit()

# --------------------------------------------------------------------------------------
# Headless Mode: Run multiple games without GUI for faster execution/testing
# --------------------------------------------------------------------------------------
def RunHeadless(num_games, agent_name, agent_name2):
    """
    Run games in headless mode (no graphics) using the Board.start() loop.
    
    In headless mode, if an agent is specified via `agent_name`, that agent will control
    Player 1 while Player 2 is controlled by our powerful agent. If no agent is provided,
    a warning is issued and both players default to the powerful agent.
    
    :param num_games: Number of games to simulate.
    :param agent_name: Name of the student's designed agent (if provided).
    """
    agent_mapping = {
        "ReflexAgent": agent_reflex,
        "Minimax": agent_minimax,
        "AlphaBeta": agent_alphabeta,
        "StrongAgent": agent_strong
    }

    if agent_name:
        if agent_name in agent_mapping:
            player1 = agent_mapping[agent_name]
        else:
            print(f"Agent '{agent_name}' not found. Exiting.")
            pygame.quit()
            sys.exit()
    else:
        player1 = None  # Human-controlled.

 
    if agent_name2:
        if agent_name2 in agent_mapping:
            player2 = agent_mapping[agent_name2]
        else:
            print(f"Agent '{agent_name2}' not found. Exiting.")
            pygame.quit()
            sys.exit()
    else:
        player2 = agent_reflex  #Default agent is reflexAgent
    agents = [player1, player2]

    results = {"Player1": 0, "Player2": 0, "Draw": 0}

    startTime = time.time()

    for i in range(num_games):
        board = Board(row=6, column=7, detail=False)
        result = board.start(agents)
        if result == 1:
            results["Player1"] += 1
        elif result == 2:
            results["Player2"] += 1
        else:
            results["Draw"] += 1
        print(f"Game {i + 1}/{num_games} finished.")

    endTime = time.time()
    executeTime = (endTime - startTime) * 1000  # 計算執行時間（毫秒）

    print(f"execute time {executeTime:.2f} ms")

    print("Summary of results:")
    print("P1 " + str(player1))
    print("P2 " + str(player2))
    print(results)
    print_INFO()

# --------------------------------------------------------------------------------------
# Main: Parse command-line arguments and launch the appropriate mode.
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Connect Four Game")
    parser.add_argument("-p", "--player", type=str, default=None,
                        help="Specify the plater 1's agent to use (e.g., ReflexAgent, Minimax, AlphaBeta, StrongAgent). ")
                             
    parser.add_argument("-e", "--enemy", type=str, default=None,
                        help="Specify the player 2's agent to use (e.g., ReflexAgent, Minimax, AlphaBeta, StrongAgent). ")
    parser.add_argument("-n", "--numgames", type=int, default=1,
                        help="Number of games to play (only applicable in headless mode).")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Run games without graphics (headless mode).")
    args = parser.parse_args()

    if args.quiet:
        RunHeadless(args.numgames, args.player, args.enemy)
    else:
        if args.numgames > 1:
            print("Multiple games (-n) are only supported in quiet mode. Running a single game in GUI mode.")
        RunGUI(agent_name=args.player, agent_name2 = args.enemy)

if __name__ == "__main__":
    main()
