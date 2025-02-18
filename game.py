'''
Oxono game
M1 Bio-info project
Daniel Calvez & Vincent Ducot
2025
'''

import sys
import datetime
import os
from model import *


USAGE = "Usage : python game.py <player1> <player2>"

def main():
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

    # Logfile name
    date = datetime.datetime.now()
    logfilename = f"logs\{date.strftime("%Y%m%d%H%M%S")}.log"
    if not os.path.exists('logs'):
        os.makedirs('logs')

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
        # Write action in log file
        with open(logfilename, mode = 'a', encoding="utf-8") as log:
            log.write(action)
            log.write("\n")

        # Check if active player wins
        print(is_winner(board, active_player, token_coord))
        if is_winner(board, active_player, token_coord):
            print(f"{active_player} gagne la partie! GG!")
            print(str_board_colored(board, player1, player2))
            sys.exit(0)
        # Check if no pawns left
        elif nb_token(board, active_player[0]+'_X') + nb_token(board, active_player[0]+'_O') + \
            nb_token(board, opponent[0]+'_X') + nb_token(board, opponent[0]+'_O') == 0:
            print("Aucun pion restant, match nul")
            print(str_board_colored(board, player1, player2))
            sys.exit(0)
        else:
            # Swap the players
            active_player, opponent = opponent, active_player

if __name__ == '__main__':
    main()
