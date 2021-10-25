# -*- coding: utf-8 -*-
"""
Author: Samuel Peters (sjpeters3@wisc.edu)
Date: 7/1/21
Connect 4

This program performs minimax search on a game of Connect 4. Connect 4 is a 
game in which players alternate turns by placing their token in a column of 
the 6x7 board. The token drops down to the lowest empty spot in that column. 
A player wins when four of their tokens form a straight line (either 
vertically, horizontally, or diagonally). The computer player uses a minimax 
algorithm with alpha beta pruning. The minimax search is limited to a depth of 
2 for time
concerns. To calculate the values once the depth limit is reached and a
terminating state has not been reached, a heuristic is used. The heuristic
prioritizes large groups of adjacent pieces of the player's tiles. Then,
the heuristic promotes certain configurations that require fewer moves to
reach a winning configuration. This is achieved by calculating the sum of
manhattan distances between all of the pieces. The worst configurations
exclusively contain some of these sums.
"""

import numpy as np
import random
import copy
import time
import math
from colorama import Fore, Style
import os

# Forces colors to work
os.system("")

# The representation of a game piece in the board
EMPTY = 0
P1_PIECE = 1
P2_PIECE = 2

# The number of consecutive pieces needed to win
CONNECT = 4

# The dimensions of the board
ROWS = 6
COLUMNS = 7

"""
Initializes and returns an empty board.
"""
def init_board():
    return np.zeros((ROWS, COLUMNS))

"""
Places the designated player piece at the specified (row, col) position. 
Assumes that the position is valid.
"""
def make_move(board, player_piece, row, col):
    board[row][col] = player_piece

"""
Returns whether or not the specified (row, col) position is empty.
"""
def valid_spot(board, row, col):
    return board[row][col] == EMPTY
"""
Returns the row of the lowest empty space in the designated column. Returns 
-1 if the entire column is full.
"""
def lowest_row(board, col):
    for row in reversed(range(ROWS)):
        if board[row][col] == EMPTY:
            return row
        
    return -1

"""
Returns a list of all possible moves that can be made on the given board.
"""
def get_valid_locations(board):
	valid_locations = []
	for col in range(COLUMNS):
		if lowest_row(board, col) > -1:
			valid_locations.append(col)
	return valid_locations

"""
Prints the board to the console.
"""
def print_board(board, latest_move):
    for row in range(ROWS):
        line = Style.RESET_ALL + str(row) + ": "
        for col in range(COLUMNS):
            cell = board[row][col]
            item = "  "
            if row == latest_move[0] and col == latest_move[1]:
                if int(cell) == 1:
                    item = Fore.RED + "X "
                elif int(cell) == 2:
                    item = Fore.RED + "O "
            else:
                if int(cell) == 1:
                    item = Style.RESET_ALL + "X "
                elif int(cell) == 2:
                    item = Style.RESET_ALL + "O "
            line += item
        print(line)
    print(Style.RESET_ALL + "   A B C D E F G\n")

"""
Checks if the specified player has a winning combination on the board. Returns 
true if the player has won, false otherwise.
"""
def check_win(board, player):
    player_piece = P1_PIECE
    if player == 1:
        player_piece = P2_PIECE
    
    # Check for horizontal wins
    for row in range(ROWS):
        for col in range(COLUMNS - CONNECT + 1):
            playerCount = 0
            for i in range(CONNECT):
                if board[row][col + i] == player_piece:
                    playerCount += 1
            if playerCount == CONNECT:
                return True
            
    # Check for vertical wins
    for col in range(COLUMNS):
        for row in range(ROWS - CONNECT + 1):
            playerCount = 0
            for i in range(CONNECT):
                if board[row + i][col] == player_piece:
                    playerCount += 1
            if playerCount == CONNECT:
                return True
            
    # Check for / diagonal wins
    for row in range(ROWS - CONNECT + 1):
        for col in range(COLUMNS - CONNECT, COLUMNS):
            playerCount = 0
            for i in range(CONNECT):
                if board[row + i][col - i] == player_piece:
                    playerCount += 1
            if playerCount == CONNECT:
                return True
            
    # Check for \ diagonal wins
    for row in range(ROWS - CONNECT + 1):
        for col in range(COLUMNS - CONNECT + 1):
            playerCount = 0
            for i in range(CONNECT):
                if board[row + i][col + i] == player_piece:
                    playerCount += 1
            if playerCount == CONNECT:
                return True
    
    # No winning combinations were found for this player
    return False

"""
Returns the heuristic score of the specified player within this CONNECT long 
window. Note that this is hard coded for connect 4.
"""
def score_window(window, player):
    # Determine the opponent's piece
    player_piece = P1_PIECE
    opp_piece = P2_PIECE
    if player == 1:
        player_piece = P2_PIECE
        opp_piece = P1_PIECE
        
    # The heuristic score of this window
    score = 0
    
    # Check for winning combination
    # Note that we want to check for the winning combination here so that the 
    # minimax algorithm chooses the "best" win condition it can find and not 
    # just any win condition.
    if window.count(player_piece) == 4:
        score += 100
    # Check for unobstructed line of 3
    elif window.count(player_piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    # Check for unobstructed line of 2
    elif window.count(player_piece) == 2 and window.count(EMPTY) == 2:
        score += 2
    # Penalize for unobstructed line of 3 for opponent pieces
    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4
    # Penalize for unobstructed line of 2 for opponent pieces
    elif window.count(opp_piece) == 2 and window.count(EMPTY) == 2:
        score -= 1

    return score

"""
Given a board and player, calculates their hueristic score. Note that while 
the hueristic score does give significant points for win conditions, the board 
should be checked for a win through check_win() before calling.
"""
def score_board(board, player):
    score = 0
    
    # Check horizontal windows
    for row in range(ROWS):
        row_array = [int(i) for i in list(board[row,:])]
        for col in range(COLUMNS - CONNECT + 1):
            window = row_array[col:col+CONNECT]
            score += score_window(window, player)
            
    # Check vertical windows
    for col in range(COLUMNS):
        col_array = [int(i) for i in list(board[:,col])]
        for row in range(ROWS - CONNECT + 1):
            window = col_array[row:row+CONNECT]
            score += score_window(window, player)
            
    # Check \ diagonal windows
    for row in range(ROWS - CONNECT + 1):
        for col in range(COLUMNS - CONNECT + 1):
            window = [board[row+CONNECT-1-i][col+i] for i in range(CONNECT)]
            score += score_window(window, player)
           
    # Check / diagonal windows
    for row in range(ROWS - CONNECT + 1):
        for col in range(COLUMNS - CONNECT + 1):
            window = [board[row+i][col+i] for i in range(CONNECT)]
            score += score_window(window, player)
        
    return score

"""
Checks if this board is terminal. A board will be terminal if either player 
has a winning combination or if the board is completely full and there is a 
draw.
"""
def is_terminal_board(board):
    return check_win(board, 0) or check_win(board, 1) or len(get_valid_locations(board)) == 0

def minimax(board, depth, max_depth, alpha, beta, maxPlayer, player):
    # Determine the opponent's piece
    player_piece = P1_PIECE
    opp_piece = P2_PIECE
    if player == 1:
        player_piece = P2_PIECE
        opp_piece = P1_PIECE
    
    # Check if this board is in a terminal state
    if is_terminal_board(board):
        # P1 win
        if check_win(board, player):
            return (None, 1000000)
        # P2 win
        elif check_win(board, (player + 1) % 2):
            return (None, -1000000)
        # Draw
        else:
            return (None, 0)
    
    # Check if we have reached the maximum tree depth
    if depth == max_depth:
        return (None, score_board(board, player))
    
    valid_locations = get_valid_locations(board)
    
    # Max player (current player)
    if maxPlayer:
        # Initialize maximum at -infinity
        value = -math.inf
        # Make initial choice random
        column = random.choice(valid_locations)
        
        # Iterate over all successors
        for col in valid_locations:
            # Get the lowest free row for this column
            row = lowest_row(board, col)
            
            # Create a copy of the board and make the move
            board_copy = copy.deepcopy(board)
            make_move(board_copy, player_piece, row, col)
            
            # Traverse the tree to the next level
            new_score = minimax(board_copy, depth + 1, max_depth, alpha, beta, False, player)[1]
            
            # If the children return a score higher than the current value, 
            # it becomes the new best move
            if new_score > value:
                value = new_score
                column = col
                
            # alpha is the max of alpha and value
            alpha = max(alpha, value)
            
            # If this alpha is greater than or equal to its parents beta, this branch is pruned
            if alpha >= beta:
                break
            
        # Return the column to drop and its heuristic score
        return column, value
    # Min player (opponent)
    else:
        # Initialize minimum at infinity
        value = math.inf
        # Make initial choice random
        column = random.choice(valid_locations)
        
        # Iterate over all successors
        for col in valid_locations:
            # Get the lowest free row for this column
            row = lowest_row(board, col)
            
            # Create a copy of the board and make the move
            board_copy = copy.deepcopy(board)
            make_move(board_copy, opp_piece, row, col)
            
            # Traverse the tree to the next level
            new_score = minimax(board_copy, depth + 1, max_depth, alpha, beta, True, player)[1]
            
            # If the children return a score lower than the current value, 
            # it becomes the new best move
            if new_score < value:
                value = new_score
                column = col
            
            # beta is the min of beta and value
            beta = min(beta, value)
            
            # If this beta is less than or equal to its parents alpha, this branch is pruned
            if beta <= alpha:
                break
        
        # Return the column to drop and its heuristic score
        return column, value

"""
Runs the game, pitting two AIs against one another
"""
def main():
    # Maximum depths for the minimax algorithm
    p1_depth = 5
    p2_depth = 1
    
    # If true, P2 will use minimax, otherwise P2 will make a random move
    p2_minimax = True
    
    # The number of games to simulate
    num_games = 100
    
    # Values used to denote which player's turn it is
    p1 = 0
    p2 = 1
    
    # Whether or not to print the final board of each game
    printing = False
    
    # Statistics to record the wins, losses, and draws of P1
    wins = 0
    losses = 0
    draws = 0
    
    start = time.perf_counter()
    
    print("Simulating", num_games, "games")
    
    while wins + losses + draws != num_games:
        
        # Start with a random player's turn
        turn = random.randint(p1, p2)
        
        # Whether or not the game has ended
        game_over = False
        
        # Initialize the board
        board = init_board()
        
        # Loop until the game is over
        while not game_over:
            # Set info for the player
            player = turn
            max_depth = p1_depth
            player_piece = P1_PIECE
            if player == 1:
                max_depth = p2_depth
                player_piece = P2_PIECE
             
            # Declare row and col here so that we can pass it to print board
            row = 0
            col = 0
            
            # AI's turn
            if player == p1:
                # Determine the move to make using the minimax algorithm
                col, minimax_score = minimax(board, 0, max_depth, -math.inf, math.inf, True, player)
                    
                # Check if the move is valid
                row = lowest_row(board, col)
                if row > -1:
                    make_move(board, player_piece, row, col)
                else:
                    print("ERROR")
            # Player's turn
            else:
                if p2_minimax:
                    # Determine the move to make using the minimax algorithm
                    col, minimax_score = minimax(board, 0, max_depth, -math.inf, math.inf, True, player)
                        
                    # Check if the move is valid
                    row = lowest_row(board, col)
                    if row > -1:
                        make_move(board, player_piece, row, col)
                    else:
                        print("ERROR")
                else:
                    valid_locations = get_valid_locations(board)
                    col = random.choice(valid_locations)
                    row = lowest_row(board, col)
                    make_move(board, player_piece, row, col)
                        
            if check_win(board, p1):
                wins += 1
                game_over = True
            elif check_win(board, p2):
                losses += 1
                game_over = True
            elif len(get_valid_locations(board)) == 0:
                draws += 1
                game_over = True
            
            turn += 1
            turn %= 2
        
        if printing:
            print_board(board, (row, col))
        print("Game ", wins + losses + draws, " / ", num_games, " complete", sep = "")
    
    print()
    print("Win ratio: ", wins, " / ", num_games, " = ", wins / num_games, sep = "")
    print("Draws:", draws)
        
    end = time.perf_counter()
    print("Average time per game: ", (end - start) / num_games, " s", sep = "")
    #print("Worst tree search amongst all games: ", worst_time, " s")
    
    input('Press ENTER to exit')
    

if __name__ == "__main__":
    main()
