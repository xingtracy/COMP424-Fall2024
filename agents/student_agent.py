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
    self.position_weights = None
    self.max_depth = 3
    # Weights for different game phases
    self.weights = {
      'opening': {
        'corner_grab': 100,
        'stability': 50,
        'mobility': 80,
        'placement': 40,
        'frontier': 30,
        'disc_diff': 0,
      },
      'midgame': {
        'corner_grab': 60,
        'stability': 50,
        'mobility': 40,
        'placement': 30,
        'frontier': 25,
        'disc_diff': 30,
      },
      'endgame': {
        'corner_grab': 30,
        'stability': 50,
        'mobility': 0,
        'placement': 20,
        'frontier': 15,
        'disc_diff': 50,
      }
    }
  
  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
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
    
    return best_move
  
  def evaluate_board(self, chess_board, player, opponent):
    """Evaluate board state with multiple heuristics"""
    board_size = chess_board.shape[0]
    empty_squares = np.sum(chess_board == 0)
    total_squares = board_size * board_size
    
    # Determine game phase
    if empty_squares > 0.7 * total_squares:
        phase = 'opening'
    elif empty_squares > 0.3 * total_squares:
        phase = 'midgame'
    else:
        phase = 'endgame'
    
    weights = self.weights[phase]
    
    # 1. Corner Grab
    corners = [(0, 0), (0, board_size-1), (board_size-1, 0), (board_size-1, board_size-1)]
    valid_moves = get_valid_moves(chess_board, player)
    corner_grab = sum(1 for move in valid_moves if move in corners)
    
    # 2. Stability
    player_stable, player_stable_board = StudentAgent.count_stable_pieces(chess_board, player)
    opponent_stable, opponent_stable_board = StudentAgent.count_stable_pieces(chess_board, opponent)
    stability = player_stable - opponent_stable
    
    # 3. Mobility
    player_moves = len(valid_moves)
    opponent_moves = len(get_valid_moves(chess_board, opponent))
    mobility = player_moves - opponent_moves
    
    # 4. Placement - my version
    # player_positions = StudentAgent.position_weights(chess_board, player_stable_board, player)
    # opponent_positions = StudentAgent.position_weights(chess_board, opponent_stable_board, opponent)
    # placement_score = player_positions - opponent_positions
    
    # 4. Placement (using position weights)
    placement_score = 0
    position_weights = self.get_position_weights(board_size)
    for i in range(board_size):
        for j in range(board_size):
            if chess_board[i][j] == player:
                placement_score += position_weights[i][j]
            elif chess_board[i][j] == opponent:
                placement_score -= position_weights[i][j]
    
    # 5. Frontier Discs
    player_frontier = StudentAgent.count_frontier_discs(chess_board, player)
    opponent_frontier = StudentAgent.count_frontier_discs(chess_board, opponent)
    frontier = opponent_frontier - player_frontier  # Fewer frontier discs is better
    
    # 6. Disc Difference
    player_discs = np.sum(chess_board == player)
    opponent_discs = np.sum(chess_board == opponent)
    disc_diff = player_discs - opponent_discs
    
    # Calculate weighted sum
    score = (
        weights['corner_grab'] * corner_grab +
        weights['stability'] * stability +
        weights['mobility'] * mobility +
        weights['placement'] * placement_score +
        weights['frontier'] * frontier +
        weights['disc_diff'] * disc_diff
    )
    
    return score

  def alpha_beta(self, chess_board, depth, alpha, beta, maximizing_player, player, opponent, start_time):
    """Minimax implementation with alpha-beta pruning and time checking"""
    
    # Time safety margin
    if time.time() - start_time > 1.97:  
      return self.evaluate_board(chess_board, player, opponent), None
      # raise TimeoutError
    
    # Base Case: At the root
    if depth == 0:
      return self.evaluate_board(chess_board, player, opponent), None
        
    is_endgame, p1_score, p2_score = check_endgame(chess_board, player, opponent)
    
    if is_endgame:
      return self.evaluate_board(chess_board, player, opponent), None
    
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
  
  def count_stable_pieces(board, player):
    """
    Count stable pieces for a specific player on a Reversi/Othello board (excluding corners).
    """
    n = len(board)
    stable = np.zeros((n, n), dtype=bool)  # Track stable pieces
    corners = {(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)}
    
    # Mark corners as stable
    for x, y in corners:
      if board[x][y] == player:
        stable[x][y] = True
    
    # Keep scanning until no new stable pieces are found
    changed = True
    while changed:
      changed = False
      for i in range(n):
        for j in range(n):
          # Skip empty squares, opponent pieces, and already stable pieces
          if board[i][j] != player or stable[i][j] or (i, j) in corners:
            continue
          
          # Check if piece is stable
          if StudentAgent.is_stable_piece(board, stable, i, j, player):
            stable[i][j] = True
            changed = True
    
    # Count stable pieces (excluding corners)
    stable_count = sum(1 for i in range(n) for j in range(n) if stable[i][j] and (i, j) not in corners)
    return stable_count, stable

  def is_stable_piece(board, stable, row, col, player):
    """
    Check if a piece is stable by verifying it's protected in all directions.
    A piece is stable if it's connected to stable pieces or board edges
    in all directions (horizontal, vertical, and both diagonals).
    """
    n = len(board)
    
    # Check all directions
    directions = [
      [(0, 1), (0, -1)],  
      [(1, 0), (-1, 0)],  
      [(1, 1), (-1, -1)], 
      [(1, -1), (-1, 1)]  
    ]
    
    # For each direction pair (e.g., left/right, up/down)
    for dir_pair in directions:
      protected = False
      # Check if protected by edge or stable pieces in either direction
      for dx, dy in dir_pair:
        x, y = row, col
        while True:
          x += dx
          y += dy
          # If we hit the edge, this direction is protected
          if x < 0 or x >= n or y < 0 or y >= n:
            protected = True
            break
          # If we hit an empty space or opponent's piece before a stable piece,
          # this direction is not protected
          if board[x][y] != player:
            break
          # If we hit a stable piece of same color, this direction is protected
          if stable[x][y]:
            protected = True
            break
        if protected:
          break 
      # If neither direction is protected, piece is not stable
      if not protected:
        return False
    return True
  
  def position_weights(board, stable_board, player):
    board_size = len(board[0])
    weights = 0
    
    # Corners are highest value
    corners = [(0, 0), (0, board_size-1), (board_size-1, 0), (board_size-1, board_size-1)]
    for x, y in corners:
      # is players piece but not in the stable board
      if stable_board[x][y]!=player and board[x][y]==player :
        weights += 100
        
    # Spaces adjacent to corners are dangerous is not in stable board
    for x, y in corners:
      for dx, dy in [(0,1), (1,0), (1,1), (-1,0), (0,-1), (-1,-1), (1,-1), (-1,1)]:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < board_size and 0 <= new_y < board_size:
          if stable_board[new_x][new_y]!=player:
            weights -= 25
                
    # Edges are valuable, not counting the ones in stable board
    for i in range(2, board_size-2):
      
      if stable_board[0][i]!=player and board[0][i]==player:
        weights += 5
        
      if stable_board[i][0]!=player and board[i][0]==player:
        weights += 5
        
      if stable_board[board_size-1][i]!=player and board[board_size-1][i]==player:
        weights += 5
        
      if stable_board[i][board_size-1]!=player and board[i][board_size-1]==player:
        weights += 5
    
    return weights

  def count_frontier_discs(board, player):
  
    """Count number of empty spaces adjacent to player's pieces"""
    directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
    frontier = 0
    board_size = board.shape[0]
    
    for i in range(board_size):
      for j in range(board_size):
        if board[i][j] == player:
          for dx, dy in directions:
            new_x, new_y = i + dx, j + dy
            if (0 <= new_x < board_size and 
              0 <= new_y < board_size and 
              board[new_x][new_y] == 0):
              frontier += 1
              break
            
    return frontier
  
  def get_position_weights(self, board_size):
    """Generate position weights for the board"""
    if self.position_weights is not None and self.position_weights.shape[0] == board_size:
        return self.position_weights
        
    weights = np.ones((board_size, board_size))
    
    # Corners are highest value
    corners = [(0, 0), (0, board_size-1), (board_size-1, 0), (board_size-1, board_size-1)]
    for x, y in corners:
        weights[x][y] = 100
        
    # Spaces adjacent to corners are dangerous
    for x, y in corners:
        for dx, dy in [(0,1), (1,0), (1,1), (-1,0), (0,-1), (-1,-1), (1,-1), (-1,1)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < board_size and 0 <= new_y < board_size:
                weights[new_x][new_y] = -25
                
    # Edges are valuable
    for i in range(2, board_size-2):
        weights[0][i] = weights[i][0] = weights[board_size-1][i] = weights[i][board_size-1] = 5
        
    self.position_weights = weights
    return weights