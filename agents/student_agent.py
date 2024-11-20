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
    start_time = time.time()
    val, move = StudentAgent.alpha_beta_move(chess_board, player, opponent, player, float('-inf'), float('inf'), start_time)
    time_taken = time.time() - start_time

    '''if time_taken > 2:
      print("depth: " +str(depth)+", num moves: "+str(num_moves))'''


    print("My AI's turn took ", time_taken, "seconds.")
    return move
    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    #return random_move(chess_board,player)
  
  POSITIONAL_WEIGHTS_6x6 = [
    [100, -20, -10, -10, -20, 100],
    [-20, -50, -2,  -2, -50, -20],
    [-10, -2,   5,   5,  -2, -10],
    [-10, -2,   5,   5,  -2, -10],
    [-20, -50, -2,  -2, -50, -20],
    [100, -20, -10, -10, -20, 100]
]

  POSITIONAL_WEIGHTS_8x8 = [
    [100, -20, 10,  5,  5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [ 10,  -2,  5,  1,  1,  5,  -2,  10],
    [  5,  -2,  1,  1,  1,  1,  -2,   5],
    [  5,  -2,  1,  1,  1,  1,  -2,   5],
    [ 10,  -2,  5,  1,  1,  5,  -2,  10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10,  5,  5, 10, -20, 100]
  ]

  POSITIONAL_WEIGHTS_10x10 = [
    [100, -20, -10,  5,   5,  5,   5, -10, -20, 100],
    [-20, -50, -2,  -2,  -2, -2,  -2,  -2, -50, -20],
    [-10,  -2,  5,   1,   1,  1,   1,   5,  -2, -10],
    [  5,  -2,  1,   1,   1,  1,   1,   1,  -2,   5],
    [  5,  -2,  1,   1,   1,  1,   1,   1,  -2,   5],
    [  5,  -2,  1,   1,   1,  1,   1,   1,  -2,   5],
    [  5,  -2,  1,   1,   1,  1,   1,   1,  -2,   5],
    [-10,  -2,  5,   1,   1,  1,   1,   5,  -2, -10],
    [-20, -50, -2,  -2,  -2, -2,  -2,  -2, -50, -20],
    [100, -20, -10,  5,   5,  5,   5, -10, -20, 100]
]
  POSITIONAL_WEIGHTS_12x12 = [
    [100, -20, -10,  5,   5,  5,   5,   5,  5, -10, -20, 100],
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
    [100, -20, -10,  5,   5,  5,   5,   5,  5, -10, -20, 100]
]


  def eval_board(board, player_color, opponent_color):
    

    #factors that influence the evaluation:
    #How many disc are the players
    #The position of the discs on the board, some cells are better than others
    #How many moves are available
    #NOT IMPLEMENTING NOW: how many discs can easily be flipped (this is a negative aspect)
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
    for row in range(row_length):
      for column in range(row_length):
        if board[row][column] == player_color:
          
            position_val += POSITIONAL_WEIGHTS[row][column]

    num_moves_left = len(get_valid_moves(board, player_color))

    result = disc_difference + position_val*2 + num_moves_left*2
    return result

  def alpha_beta_move(board, player_color, opponent_color, maximize_player_color,  alpha, beta, start_time):
    if time.time() - start_time >=2:
      return StudentAgent.eval_board(board, player_color, opponent_color), None
    valid_moves = get_valid_moves(board, player_color)
    if len(valid_moves) == 0:
      return StudentAgent.eval_board(board, player_color, opponent_color), None
    
    #need to implement minimax
    if maximize_player_color == player_color:
      max_val = float('-inf')
      final_move = None
      for move in valid_moves:
        temp_board = board.copy()
        #need to check if execute move actually changes the board
        execute_move(temp_board, move, player_color)
        val, _ = StudentAgent.alpha_beta_move(temp_board, opponent_color, player_color, maximize_player_color,  alpha, beta)
        if val > max_val:
          max_val = val
          final_move = move
        
        #do far did minimax now implement alpha beta
        #in max branch need to get max val
        alpha = max(alpha, val)
        if alpha >= beta:
          break
      return max_val, final_move
    else:#minimize turn
      min_val = float('inf')
      final_move = None
      for move in valid_moves:
        temp_board = board.copy()
        #need to check if execute move actually changes the board
        execute_move(temp_board, move, player_color)
        val, _ = StudentAgent.alpha_beta_move(temp_board, opponent_color, player_color, maximize_player_color,  alpha, beta)
        if val < min_val:
          min_val = val
          final_move = move
        
        #do far did minimax now implement alpha beta
        #in max branch need to get max val
        beta = min(beta, val)
        if alpha >= beta:
          break
      return min_val, final_move
  

#result agaisnt greedy is alwasy the same since no random factor