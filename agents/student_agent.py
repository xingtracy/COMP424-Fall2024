# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves
import subprocess

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
    # Adjustable based on board size
    self.max_depth = 3
  
  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """
    start_time = time.time()
    board_size = chess_board.shape[0]
    empty_squares = np.sum(chess_board == 0)
    total_squares = board_size * board_size
    
    # Max depth based on board size and game phase
    if board_size == 6:
      self.max_depth = 5
    elif board_size == 8:
      self.max_depth = 4
    elif board_size == 10:
      self.max_depth = 3
    else:
      self.max_depth = 2
        
    # Increase depth in endgame
    if empty_squares < total_squares / 4:
      self.max_depth += 1
        
    try:
      _, best_move = self.alpha_beta(
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

  # def initialize_weights(self, board_size):
  #   """Initialize position weights for the given board size"""
  #   weights = np.ones((board_size, board_size))
    
  #   # Corners are very valuable: weight of 5
  #   weights[0, 0] = weights[0, board_size-1] = 10
  #   weights[board_size-1, 0] = weights[board_size-1, board_size-1] = 10
    
  #   # # Positions adjacent to corners are dangerous: weight of -2
  #   # weights[0, 1] = weights[1, 0] = weights[1, 1] = -2
  #   # weights[0, board_size-2] = weights[1, board_size-2] = weights[1, board_size-1] = -2
  #   # weights[board_size-2, 0] = weights[board_size-2, 1] = weights[board_size-1, 1] = -2
  #   # weights[board_size-2, board_size-1] = weights[board_size-2, board_size-2] = weights[board_size-1, board_size-2] = -2
    
  #   # # The rest of the edges are good: weight of 2
  #   # weights[0, 2:board_size-2] = 2
  #   # weights[board_size-1, 2:board_size-2] = 2
  #   # weights[2:board_size-2, 0] = 2
  #   # weights[2:board_size-2, board_size-1] = 2
    
    
  # return weights

  def find_good_edges( matrix, num):
    
    n = len(matrix)  
    result = []  
    corners = [(0, 0), (n - 1, 0),(n - 1, n - 1),(0, n - 1),]
    
    # Helper function to check edge connected to a corner
    def check_edge(corner):
      x, y = corner
      coords = []
      
      #top left
      if x == 0 and y == 0 :  
          # Top left to top right 
          for col in range(n):
              if matrix[x][col] == num:
                  coords.append((x, col))
              else:
                  break 
          # Top left to bottom left
          for row in range(n):
              if matrix[row][y] == num:
                  coords.append((row, y))
              else:
                  break
      # Bottom left
      if x == (n - 1) and y == 0:  
        # Bottom left to bottom right
        for col in range(n):
          if matrix[x][col] == num:
            coords.append((x, col))
          else:
            break
        # Bottom left to Top left
        for row in range(n):
          if matrix[-(row+1)][y] == num:
            coords.append(((n-1)-row, y))
          else:
            break
              
      # Bottom right
      if x == (n - 1) and y == (n - 1):  
        # Bottom right to bottom left
        for col in range(n):
          if matrix[x][-(1+col)] == num:
            coords.append((x, (n-1)-col))
          else:
            break
        # Bottom right to Top right
        for row in range(n):
          if matrix[-(row+1)][y] == num:
            coords.append(((n-1)-row, y))
          else:
            break
          
      # Top right
      if x == 0 and y == (n - 1):       
        # Top right to bottom left
        for row in range(n):
          if matrix[row][y] == num:
            coords.append((row, y))
          else:
            break
        # Top right to Top left
        for col in range(n):
          if matrix[x][-(1+col)] == num:
            coords.append((x, (n-1)-col))
          else:
            break
      unique_coords=list(set(coords))
      return unique_coords
    
    # Iterate through each corner
    for corner in corners:
      if matrix[corner[0]][corner[1]] == num:
        result.extend(check_edge(corner))
    
    # Remove corners
    result = [item for item in result if item not in corners]
    
    # Remove duplicates and return as a list
    return list(set(result)) 

  def evaluate_board(self, chess_board, player, opponent):
    """Evaluate board state"""
    # if self.position_weights is None:
    #     self.position_weights = self.initialize_weights(chess_board.shape[0])
    
    # player_edges = StudentAgent.find_good_edges(chess_board,player)
    # opponent_edges = StudentAgent.find_good_edges(chess_board,opponent)
    
    # weight_edges_p = sum(1 for x, y in player_edges if chess_board[x, y] == player)
    # weight_edges_o = sum(1 for x, y in opponent_edges if chess_board[x, y] == opponent)
    
    # # Count pieces with position weights
    # player_score = np.sum(np.where(chess_board == player, self.position_weights, 0))
    # opponent_score = np.sum(np.where(chess_board == opponent, self.position_weights, 0))
    
    # # Count mobility (number of valid moves)
    # player_mobility = len(get_valid_moves(chess_board, player))
    # opponent_mobility = len(get_valid_moves(chess_board, opponent))
    
    # # Combine factors
    # return (player_score - opponent_score) + 5 * (player_mobility - opponent_mobility) + 5 * (weight_edges_p-weight_edges_o)
    
    player_edges = StudentAgent.find_good_edges(chess_board,player)
    opponent_edges = StudentAgent.find_good_edges(chess_board,opponent)
    
    player_edges = sum(1 for x, y in player_edges if chess_board[x, y] == player)
    opponent_edges = sum(1 for x, y in opponent_edges if chess_board[x, y] == opponent)
    
     # Count total pieces
    player_score = np.sum(chess_board == player)
    opponent_score = np.sum(chess_board == opponent)

    # Corner control
    corners = [
      (0, 0), (0, chess_board.shape[0] - 1),
      (chess_board.shape[0] - 1, 0), (chess_board.shape[0] - 1, chess_board.shape[0] - 1)
    ]
    
    player_corners = sum(1 for x, y in corners if chess_board[x, y] == player)
    opponent_corners = sum(1 for x, y in corners if chess_board[x, y] == opponent)

    # Mobility (valid moves)
    player_moves = len(get_valid_moves(chess_board, player))
    opponent_moves = len(get_valid_moves(chess_board, opponent))

    # Weighted evaluation
    score = (
      c * (player_corners - opponent_corners) + 
      m * (player_moves - opponent_moves) +      
      s * (player_score - opponent_score) +          
      e * (player_edges - opponent_edges)
    )
     
    return score

  def alpha_beta(self, chess_board, depth, alpha, beta, maximizing_player, player, opponent, start_time):
    """Minimax implementation with alpha-beta pruning and time checking"""
    
    # Time safety margin
    if time.time() - start_time > 1.98:  
      return self.evaluate_board(chess_board, player, opponent), None
      # raise TimeoutError
    
    # Base Case: At the root
    if depth == 0:
      return self.evaluate_board(chess_board, player, opponent), None
        
    is_endgame, p1_score, p2_score = check_endgame(chess_board, player, opponent)
    
    if is_endgame:
      # score = p1_score - p2_score if player == 1 else p2_score - p1_score
      # High weight for winning positions
      return self.evaluate_board(chess_board, player, opponent), None
      # return score * 1000, None  
        
    current_player = player if maximizing_player else opponent
    other_player = opponent if maximizing_player else player
    valid_moves = get_valid_moves(chess_board, current_player)
    
    if not valid_moves:
      # If no moves, pass turn
      return self.alpha_beta(chess_board, depth-1, alpha, beta, not maximizing_player, player, opponent, start_time)[0], None
    
    # Initialize 
    best_move = valid_moves[0]
    best_value = float('-inf') if maximizing_player else float('inf')
    
    for move in valid_moves:
      board_copy = deepcopy(chess_board)
      execute_move(board_copy, move, current_player)
      
      value, _ = self.alpha_beta(board_copy, depth-1, alpha, beta, not maximizing_player, player, opponent, start_time)
      
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

  

CORNER_WEIGHTS=[10,8,5]
EDGES_WEIGHTS=[10,8,5]
MOBILITY_WEIGHTS=[5,3,2]
SCORE_WEIGHTS=[3,2,1]

autoplay_num=20
board_size=10

for c in CORNER_WEIGHTS:
  for e in EDGES_WEIGHTS:
    for m in MOBILITY_WEIGHTS:
      for s in SCORE_WEIGHTS:
        print("Corner weight of "+c)
        print("Edges weight of "+e)
        print("Mobility weight of "+m)
        print("Corner weight of "+s)
        command = "python3 simulator.py --player_1 student_agent --player_2 richard --autoplay --autoplay_runs "+ autoplay_num+" --board_size "+board_size
        result = subprocess.run(command, capture_output=True, text=True)
        print(result)