# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("richard") 
class Richard(Agent):
    """
    A class for your implementation of the agent using Alpha-Beta Pruning.
    """

    def __init__(self):
        super(Richard, self).__init__()
        self.name = "Richard"
        self.max_depth = 3  # Depth limit for Alpha-Beta search

    def step(self, chess_board, player, opponent):
        """
        Implements the step function for the AI agent.
        Uses Alpha-Beta Pruning to select the best move.
        """
        start_time = time.time()
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        # Get all valid moves for the current player
        valid_moves = get_valid_moves(chess_board, player)
        if not valid_moves:
            return None  # No valid moves available

        # Iterate through each move to find the best one
        for move in valid_moves:
            # Create a copy of the board and simulate the move
            next_board = deepcopy(chess_board)
            execute_move(next_board, move, player)

            # Call Alpha-Beta search for the opponent's turn
            score = self.alpha_beta(
                next_board, self.max_depth - 1, alpha, beta, False, player, opponent, start_time
            )

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, best_score)

        print("My AI's turn took", time.time() - start_time, "seconds.")
        return best_move

    def alpha_beta(self, board, depth, alpha, beta, maximizing_player, player, opponent, start_time):
        """
        Alpha-Beta Pruning implementation with depth limit and time check.
        """
        # Check time limit to avoid exceeding 2 seconds
        if time.time() - start_time > 1.9:
            return self.evaluate(board, player, opponent)

        # Check if the game has ended or depth limit is reached
        is_endgame, player_score, opponent_score = check_endgame(board, player, opponent)
        if is_endgame or depth == 0:
            return self.evaluate(board, player, opponent)

        # Get the valid moves for the current player
        current_player = player if maximizing_player else opponent
        valid_moves = get_valid_moves(board, current_player)

        if not valid_moves:
            # No valid moves for the current player; switch to the other player
            return self.alpha_beta(board, depth, alpha, beta, not maximizing_player, player, opponent, start_time)

        if maximizing_player:
            max_eval = float('-inf')
            for move in valid_moves:
                next_board = deepcopy(board)
                execute_move(next_board, move, player)
                eval = self.alpha_beta(next_board, depth - 1, alpha, beta, False, player, opponent, start_time)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                next_board = deepcopy(board)
                execute_move(next_board, move, opponent)
                eval = self.alpha_beta(next_board, depth - 1, alpha, beta, True, player, opponent, start_time)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate(self, board, player, opponent):
        """
        Heuristic evaluation function for the board state.
        Scores are positive for the `player` and negative for the `opponent`.
        """
        # Count total pieces
        player_score = np.sum(board == player)
        opponent_score = np.sum(board == opponent)

        # Corner control
        corners = [
            (0, 0), (0, board.shape[0] - 1),
            (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[0] - 1)
        ]
        player_corners = sum(1 for x, y in corners if board[x, y] == player)
        opponent_corners = sum(1 for x, y in corners if board[x, y] == opponent)

        # Mobility (valid moves)
        player_moves = len(get_valid_moves(board, player))
        opponent_moves = len(get_valid_moves(board, opponent))

        # Weighted evaluation
        score = (
            10 * (player_corners - opponent_corners) +  # Corner control
            5 * (player_moves - opponent_moves) +       # Mobility
            (player_score - opponent_score)            # Total piece count
        )
        return score


