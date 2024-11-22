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

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    # num_moves = len(get_valid_moves(chess_board, player))
    #depth = StudentAgent.find_depth(len(chess_board[0]), num_moves)
    #depth = StudentAgent.determine_optimal_depth(chess_board, player)
    StudentAgent.TIME_ENDED = -1
    num_pieces = np.count_nonzero(chess_board)
    num_moves = len(get_valid_moves(chess_board, player))
    start_time = time.time()
    val, move, visited, time_ended = StudentAgent.alpha_beta_move(chess_board, player, opponent, player, float('-inf'), float('inf'), start_time, 0,-1)
    end_time = time.time()
    time_taken = end_time - start_time

    StudentAgent.record_to_csv("data_records.cvs",time_taken,player,opponent,move,val)

    if time_taken > 2:
      print("My AI's TOOK OVER 2 SECONDS ", time_taken, "seconds.")


    #print("My AI's turn took ", time_taken, "seconds.")
    return move
    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    #return random_move(chess_board,player)
  def find_good_edges( matrix, num):
    n = len(matrix)  
    result = []  

    corners = [
        (0, 0), 
        (n - 1, 0),
        (n - 1, n - 1),
        (0, n - 1),
    ]
    
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
    
  def record_to_csv(file_name, *args):
    """
    Appends input data to a CSV file without importing any libraries.

    Parameters:
    - file_name: The name of the CSV file (str).
    - *args: Values to write as a row in the CSV file.
    """
    row = ','.join(map(str, args)) + '\n'

    with open(file_name, mode='a', encoding='utf-8') as file:
        file.write(row)
  
  def eval_board(board, player_color, opponent_color):
    """Call this function at leaves of the alpha beta tree to 
    evaluate "likelyhood" for our player to win

    Args:
        board (Array[Array[int]])
        player_color (int): 1 for player 1 and 2 for player 2
        opponent_color (int): 1 for player 1 and 2 for player 2

    Returns:
      int: How good of a position is our player to win based on the board
    """
    
    # Factors that influence the evaluation:
    
    # - substract weight in opponent weights as well -> Done!
    # - substract weight of pieces that are easy to flip -> In Progress
    # - add weight for pieces connected to corner -> Done!
    
    count_player = np.count_nonzero(board == player_color)
    count_opponent = np.count_nonzero(board == opponent_color)
    disc_difference = count_player - count_opponent
    
    row_length = len(board[0])
    POSITIONAL_WEIGHTS = []
    if row_length == 6:
      POSITIONAL_WEIGHTS = StudentAgent.POSITIONAL_WEIGHTS_6x6
    elif row_length == 8:
      POSITIONAL_WEIGHTS = StudentAgent.POSITIONAL_WEIGHTS_8x8
    elif row_length == 10:
      POSITIONAL_WEIGHTS = StudentAgent.POSITIONAL_WEIGHTS_10x10
    elif row_length == 12:
      POSITIONAL_WEIGHTS = StudentAgent.POSITIONAL_WEIGHTS_12x12
    position_val = 0
    
    good_edges = StudentAgent.find_good_edges(board,player_color)
    bad_edges = StudentAgent.find_good_edges(board,opponent_color)
    
    for row in range(row_length):
      for column in range(row_length):
        if (row,column) in good_edges.append(bad_edges):
          POSITIONAL_WEIGHTS[row][column]=30
        if board[row][column] == player_color:
          position_val += POSITIONAL_WEIGHTS[row][column]
        if board[row][column] == opponent_color:
          position_val -= POSITIONAL_WEIGHTS[row][column]
            
    w_disc_difference=1       
    w_postition_cal=2
    w_num_moves_left=2
    w_num_moves_opponent=1
    
    num_moves_left = len(get_valid_moves(board, player_color))
    num_moves_opponent = len(get_valid_moves(board, opponent_color))
    
    
    result = (disc_difference*w_disc_difference) + (position_val*w_postition_cal) + (num_moves_left*w_num_moves_left) - (num_moves_opponent*w_num_moves_opponent)
    
    # Record the weights we tried to see which gives the better result
    StudentAgent.record_to_csv("weight_tunning.csv", w_disc_difference,w_postition_cal,w_num_moves_left,w_num_moves_opponent,result)
    
    return result

  def alpha_beta_move(board, player_color, opponent_color, maximize_player_color, alpha, beta, start_time, num_vodes_visited, time_ended):
    """
      Based on the state, we want to find move to maximize the player to win under 1.95s

    Args:
        board (Array[Array[int]])
        player_color (int): 1 for player 1 and 2 for player 2
        opponent_color (int): 1 for player 1 and 2 for player 2
        maximize_player_color (int): 1 for player 1 and 2 for player 2
        alpha (float): 
        beta (float): 
        start_time (time):
        num_vodes_visited (int):
        time_ended (time):
    Returns:
        min_val, final_move, num_vodes_visited, time_ended: 
    """
    
    end = time.time()
    
    if end - start_time >= 1.99:# or num_vodes_visited >= 11000:
      if time_ended == -1:
          time_ended = end
          
      return StudentAgent.eval_board(board, player_color, opponent_color), None, num_vodes_visited, time_ended
    
    valid_moves = get_valid_moves(board, player_color)
    
    # No valid moves
    if len(valid_moves) == 0:
      return StudentAgent.eval_board(board, player_color, opponent_color), None, num_vodes_visited, time_ended
    
    # Maximize turn
    if maximize_player_color == player_color:
      # if time.time() - start_time >=1.85:
      #   return StudentAgent.eval_board(board, player_color, opponent_color), None
      
      max_val = float('-inf')
      final_move = None
      
      for move in valid_moves:
        
        temp_board = board.copy()
        
        # Check if execute move actually changes the board
        execute_move(temp_board, move, player_color)
        num_vodes_visited+= 1
        val, _,  new_num_nodes, new_time= StudentAgent.alpha_beta_move(temp_board, opponent_color, player_color, maximize_player_color,  alpha, beta, start_time, num_vodes_visited, time_ended)
        num_vodes_visited = new_num_nodes
        
        if time_ended == -1 and new_time != -1:
          time_ended = new_time
        
        if val > max_val:
          max_val = val
          final_move = move
        
        #do far did minimax now implement alpha beta
        #in max branch need to get max val
        alpha = max(alpha, val)
        if alpha >= beta:
          break
        
        if time.time() - start_time >= 1.99:# or num_vodes_visited >= 11000:
            break
      
      return max_val, final_move, num_vodes_visited, time_ended
    
    # Minimize turn
    else:
      
      # if time.time() - start_time >=1.90:
      #   return StudentAgent.eval_board(board, player_color, opponent_color), None
      
      min_val = float('inf')
      final_move = None
      
      for move in valid_moves:
        
        temp_board = board.copy()
        
        # Check if execute move actually changes the board
        execute_move(temp_board, move, player_color)
        num_vodes_visited+= 1
        val, _, new_num_nodes, new_time = StudentAgent.alpha_beta_move(temp_board, opponent_color, player_color, maximize_player_color,  alpha, beta, start_time, num_vodes_visited, time_ended)
        num_vodes_visited = new_num_nodes
        
        if time_ended == -1 and new_time != -1:
            time_ended = new_time
        
        if val < min_val:
          min_val = val
          final_move = move
        
        #do far did minimax now implement alpha beta
        #in max branch need to get max val
        beta = min(beta, val)
        if alpha >= beta:
          break
        
        if time.time() - start_time >= 1.99:# or num_vodes_visited >= 11000:
            break
      
      return min_val, final_move, num_vodes_visited, time_ended
    
  
  POSITIONAL_WEIGHTS_6x6 = [
    [50, -20, -10, -10, -20, 50],
    [-20, -50, -2,  -2, -50, -20],
    [-10, -2,   5,   5,  -2, -10],
    [-10, -2,   5,   5,  -2, -10],
    [-20, -50, -2,  -2, -50, -20],
    [50, -20, -10, -10, -20, 50]
  ]
  POSITIONAL_WEIGHTS_8x8 = [
    [50, -20, 10,  5,  5, 10, -20, 50],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [ 10,  -2,  5,  1,  1,  5,  -2,  10],
    [  5,  -2,  1,  1,  1,  1,  -2,   5],
    [  5,  -2,  1,  1,  1,  1,  -2,   5],
    [ 10,  -2,  5,  1,  1,  5,  -2,  10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [50, -20, 10,  5,  5, 10, -20, 50]
  ]
  POSITIONAL_WEIGHTS_10x10 = [
    [50, -20, -10,  5,   5,  5,   5, -10, -20, 50],
    [-20, -50, -2,  -2,  -2, -2,  -2,  -2, -50, -20],
    [-10,  -2,  5,   1,   1,  1,   1,   5,  -2, -10],
    [  5,  -2,  1,   1,   1,  1,   1,   1,  -2,   5],
    [  5,  -2,  1,   1,   1,  1,   1,   1,  -2,   5],
    [  5,  -2,  1,   1,   1,  1,   1,   1,  -2,   5],
    [  5,  -2,  1,   1,   1,  1,   1,   1,  -2,   5],
    [-10,  -2,  5,   1,   1,  1,   1,   5,  -2, -10],
    [-20, -50, -2,  -2,  -2, -2,  -2,  -2, -50, -20],
    [50, -20, -10,  5,   5,  5,   5, -10, -20, 50]
  ]
  POSITIONAL_WEIGHTS_12x12 = [
    [50, -20, -10,  5,   5,  5,   5,   5,  5, -10, -20, 50],
    [-20, -50, -2,  -2,  -2, -2,  -2,  -2, -2,  -2, -50, -20],
    [-10,  -2,  5,   1,   1,  1,   1,   1,  1,   1,  -2, -10],
    [  5,  -2,  1,   1,   1,  1,   1,   1,  1,   1,  -2,   5],
    [  5,  -2,  1,   1,   1,  1,   1,   1,  1,   1,  -2,   5],
    [  5,  -2,  1,   1,   1,  1,   1,   1,  1,   1,  -2,   5],
    [  5,  -2,  1,   1,   1,  1,   1,   1,  1,   1,  -2,   5],
    [  5,  -2,  1,   1,   1,  1,   1,   1,  1,   1,  -2,   5],
    [-10,  -2,  5,   1,   1,  1,   1,   1,  1,   1,  -2, -10],
    [-20, -50, -2,  -2,  -2, -2,  -2,  -2, -2,  -2, -50, -20],
    [-20, -50, -2,  -2,  -2, -2,  -2,  -2, -2,  -2, -50, -20],
    [50, -20, -10,  5,   5,  5,   5,   5,  5, -10, -20, 50]
  ]




  '''
  def find_depth(size, num_moves):
    max_values = [9, 5, 4, 4]
    max_val = 3
    if size == 6:
      max_val = max_values[0]
      if num_moves == 4:
        max_val = 7
      if num_moves >= 5:
        max_val =6
    elif size == 8:
      max_val = max_values[1]
      if num_moves >=11:
        max_val -= 1

    elif size == 10:
      max_val = max_values[2]
      if num_moves >= 15:
        max_val -= 1

    elif size == 12:
      max_val = max_values[3]
      if num_moves >= 13:
        max_val -= 1

    return max_val
    
  def determine_optimal_depth(board, player_color):
    num_moves = len(get_valid_moves(board, player_color))
    total_pieces = np.count_nonzero(board)
    board_size = len(board)  # Determine the size of the board (6, 8, 10, or 12)

    if board_size == 6:
        # Heuristic thresholds for 6x6 board
        if total_pieces < 10:  # Early game
            if num_moves < 5:
                return 7
            else:
                return 6
        elif total_pieces < 20:  # Mid game
            if num_moves < 10:
                return 6
            else:
                return 5
        else:  # Late game
            if num_moves < 5:
                return 8
            else:
                return 6
    elif board_size == 8:
        # Heuristic thresholds for 8x8 board
        if total_pieces < 20:  # Early game
            if num_moves < 10:
                return 6
            else:
                return 5
        elif total_pieces < 40:  # Mid game
            if num_moves < 15:
                return 5
            else:
                return 4
        else:  # Late game
            if num_moves < 10:
                return 6
            else:
                return 5
    elif board_size == 10:
        # Heuristic thresholds for 10x10 board
        if total_pieces < 30:  # Early game
            if num_moves < 15:
                return 5
            else:
                return 4
        elif total_pieces < 50:  # Mid game
            if num_moves < 20:
                return 4
            else:
                return 3
        else:  # Late game
            if num_moves < 15:
                return 5
            else:
                return 4
    elif board_size == 12:
        # Heuristic thresholds for 12x12 board
        if total_pieces < 40:  # Early game
            if num_moves < 20:
                return 4
            else:
                return 3
        elif total_pieces < 60:  # Mid game
            if num_moves < 25:
                return 3
            else:
                return 2
        else:  # Late game
            if num_moves < 20:
                return 4
            else:
                return 3
    else:
        raise ValueError("Unsupported board size")
  '''