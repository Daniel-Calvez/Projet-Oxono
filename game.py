'''
Oxono game
M1 Bio-info project
Daniel Calvez & Vincent Ducot
2025
'''

import sys
from model import *


USAGE = "Usage : python game.py <player1> <player2>"

if len(sys.argv) < 3:
    print(USAGE)
    sys.exit(1)

player1 = sys.argv[1]
player2 = sys.argv[2]

if not valid_player_names(player1, player2):
    print("Les noms des joueurs ne sont pas valides")
    sys.exit(1)

# Initialize the board
board = init_board()
set_player1(player1)
active_player = player1
opponent = player2

# Main loop
while True:
    # Get the action from the player
    action = ask_play(board, active_player, opponent)
    # Split the action string
    if action[0] == 'X':
        token = active_player[0]+'_X'
        totem = 'T_X'
    else:
        token = active_player[0]+'_O'
        totem = 'T_O'
    totem_coord = convert_coord(action[1:3])
    token_coord= convert_coord(action[3:5])
    # Move totem
    move_totem(board, totem, totem_coord)
    # Move pawn
    board[token_coord[0]][token_coord[1]] = token
    # Check if active player wins
    if is_winner(board, active_player, token_coord):
        print(f"{active_player} gagne la partie! GG!")
        sys.exit(0)
    # Check if no pawns left
    elif nb_token(board, active_player[0]+'_X') + nb_token(board, active_player[0]+'_O') + \
        nb_token(board, opponent[0]+'_X') + nb_token(board, opponent[0]+'_O') == 0:
        print("Aucun pion restant, match nul")
        sys.exit(0)
    else:
        tmp = active_player
        active_player = opponent
        opponent = active_player
