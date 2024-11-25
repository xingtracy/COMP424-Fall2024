# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
"""
  def __init__(self):
      super(StudentAgent, self).__init__()
      self.name = "StudentAgent"
      # Weights for board position evaluation
      self.position_weights = None
      self.max_depth = 4  # Adjustable based on board size

  def initialize_weights(self, board_size):
      """Initialize position weights for the given board size"""
      weights = np.ones((board_size, board_size))
      
      # Corners are highly valuable
      weights[0, 0] = weights[0, board_size-1] = 5
      weights[board_size-1, 0] = weights[board_size-1, board_size-1] = 5
      
      # Positions adjacent to corners are dangerous
      weights[0, 1] = weights[1, 0] = weights[1, 1] = -2
      weights[0, board_size-2] = weights[1, board_size-2] = weights[1, board_size-1] = -2
      weights[board_size-2, 0] = weights[board_size-2, 1] = weights[board_size-1, 1] = -2
      weights[board_size-2, board_size-1] = weights[board_size-2, board_size-2] = weights[board_size-1, board_size-2] = -2
      
      # Edges are good
      weights[0, 2:board_size-2] = 2
      weights[board_size-1, 2:board_size-2] = 2
      weights[2:board_size-2, 0] = 2
      weights[2:board_size-2, board_size-1] = 2
      
      return weights

  def evaluate_board(self, chess_board, player, opponent):
      """Evaluate the board state"""
      if self.position_weights is None:
          self.position_weights = self.initialize_weights(chess_board.shape[0])
          
      # Count pieces with position weights
      player_score = np.sum(np.where(chess_board == player, self.position_weights, 0))
      opponent_score = np.sum(np.where(chess_board == opponent, self.position_weights, 0))
      
      # Count mobility (number of valid moves)
      player_mobility = len(get_valid_moves(chess_board, player))
      opponent_mobility = len(get_valid_moves(chess_board, opponent))
      
      # Combine factors
      return player_score - opponent_score + 0.5 * (player_mobility - opponent_mobility)

  def minimax(self, chess_board, depth, alpha, beta, maximizing_player, player, opponent, start_time):
      """Minimax implementation with alpha-beta pruning and time checking"""
      if time.time() - start_time > 1.95:  # Time safety margin
          raise TimeoutError
          
      if depth == 0:
          return self.evaluate_board(chess_board, player, opponent), None
          
      is_endgame, p1_score, p2_score = check_endgame(chess_board, player, opponent)
      if is_endgame:
          score = p1_score - p2_score if player == 1 else p2_score - p1_score
          return score * 1000, None  # High weight for winning positions
          
      current_player = player if maximizing_player else opponent
      other_player = opponent if maximizing_player else player
      valid_moves = get_valid_moves(chess_board, current_player)
      
      if not valid_moves:
          # If no moves, pass turn
          return self.minimax(chess_board, depth-1, alpha, beta, not maximizing_player, player, opponent, start_time)[0], None
          
      best_move = valid_moves[0]
      best_value = float('-inf') if maximizing_player else float('inf')
      
      for move in valid_moves:
          board_copy = deepcopy(chess_board)
          execute_move(board_copy, move, current_player)
          
          value, _ = self.minimax(board_copy, depth-1, alpha, beta, not maximizing_player, player, opponent, start_time)
          
          if maximizing_player:
              if value > best_value:
                  best_value = value
                  best_move = move
              alpha = max(alpha, best_value)
          else:
              if value < best_value:
                  best_value = value
                  best_move = move
              beta = min(beta, best_value)
              
          if beta <= alpha:
              break
              
      return best_value, best_move

  def step(self, chess_board, player, opponent):
      """Main step function"""
      start_time = time.time()
      
      # Adjust depth based on board size and game phase
      board_size = chess_board.shape[0]
      empty_squares = np.sum(chess_board == 0)
      total_squares = board_size * board_size
      
      # Adaptive depth based on board size and game phase
      if board_size <= 6:
          self.max_depth = 5
      elif board_size <= 8:
          self.max_depth = 4
      else:
          self.max_depth = 3
          
      # Increase depth in endgame
      if empty_squares < total_squares / 4:
          self.max_depth += 1
          
      try:
          _, best_move = self.minimax(
              chess_board,
              self.max_depth,
              float('-inf'),
              float('inf'),
              True,
              player,
              opponent,
              start_time
          )
          
      except TimeoutError:
          # If we timeout, return the best move found so far
          valid_moves = get_valid_moves(chess_board, player)
          if valid_moves:
              best_move = valid_moves[0]
              # Quick evaluation of immediate moves
              best_score = float('-inf')
              for move in valid_moves:
                  board_copy = deepcopy(chess_board)
                  execute_move(board_copy, move, player)
                  score = self.evaluate_board(board_copy, player, opponent)
                  if score > best_score:
                      best_score = score
                      best_move = move
          else:
              best_move = None
              
      time_taken = time.time() - start_time
      if time_taken > 2:
        print("My AI's TOOK OVER 2 SECONDS ", time_taken, "seconds.")
      # print(f"Move took {time_taken:.3f} seconds")
      
      return best_move

