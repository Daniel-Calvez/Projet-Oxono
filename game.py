'''
Oxono game
M1 Bio-info project
Daniel Calvez & Vincent Ducot
2025
'''

import sys
import datetime
import os
import time
import argparse
import textwrap
import model
import ia


def ia_play(board: list[list[str]], active_player: str, opponent: str, ia_level: int) -> str:
    '''
    Call the IA to play a move. Verify constraints of 1 second to answer and 3 tries.
    Args 
        The board as a matrix
        Current player's name
        Opponent's name
        IA level, choice in (0,1,2)
    Returns
        The action to play as a string
    Exception
        No exception
    '''
    valid_action = False
    action = ""
    # Get the action from the IA, ask only 3 times
    ia_counter = 3
    print(model.str_board_colored(board))
    while not valid_action and ia_counter > 0:
        # One second to answer
        start = time.time()
        action = ia.ask_play(board, active_player, opponent, ia_level)
        end = time.time()
        if end - start > 1:
            print(f"IA {active_player} a été trop longue à répondre, disqualifiée!")
            sys.exit(0)
        valid_action = model.is_valid_action(board, action, active_player)
        ia_counter -= 1
    if ia_counter == 0:
        print(f"IA {active_player} n'a pas répondu en 3 essais, disqualifiée!")
        sys.exit(0)

    return action


def main(args):
    '''
    Game's main loop
    Check the args then loop until the game is over.
    '''
    player_vs_ia = False
    ia_vs_ia = False

    player1 = args.player1
    player2 = args.player2

    # Check if there is any IA
    if args.player1_ia:
        if args.player2_ia:
            print(f"Le 1er joueur est une IA appelée {player1}")
            ia_vs_ia = True
        else:
            print("Le 1er joueur ne peut pas être une IA si le second joueur n'est pas une IA.")
            sys.exit(1)

    if args.player2_ia:
        print(f"Le 2e joueur est une IA appelée {ia.name()}")
        player2 = ia.name()
        player_vs_ia = True

    if not model.valid_player_names(player1, player2):
        print("Les noms des joueurs ne sont pas valides")
        sys.exit(1)

    # Initialize the board
    board = model.init_board()
    model.set_player1(player1)
    active_player = player1
    opponent = player2

    # Logfile name
    date = datetime.datetime.now()
    logfilename = os.path.join('logs', date.strftime("%Y%m%d%H%M%S") + '.log')
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Main loop
    while True:
        # Get action from IA
        if (player_vs_ia and active_player == player2) or ia_vs_ia:
            if active_player == player1:
                action = ia_play(board, active_player, opponent, args.ia1_level)
            else:
                action = ia_play(board, active_player, opponent, args.ia2_level)
        # Get the action from the player
        else:
            action = model.ask_play(board, active_player, opponent)

        # Split the action string
        if action[0] == 'X':
            token = active_player[0]+'_X'
            totem = 'T_X'
        else:
            token = active_player[0]+'_O'
            totem = 'T_O'
        totem_coord = model.convert_coord(action[1:3])
        token_coord= model.convert_coord(action[3:5])
        # Move totem
        model.move_totem(board, totem, totem_coord)
        # Move pawn
        board[token_coord[0]][token_coord[1]] = token
        # Write action in log file
        with open(logfilename, mode = 'a', encoding="utf-8") as log:
            log.write(action)
            log.write("\n")

        # Check if active player wins
        if model.is_winner(board, active_player, token_coord):
            print(f"{active_player} gagne la partie! GG!")
            print(model.str_board_colored(board))
            sys.exit(0)
        # Check if no pawns left
        elif model.nb_token(board, active_player[0]+'_X') + model.nb_token(board, active_player[0]+'_O') + \
            model.nb_token(board, opponent[0]+'_X') + model.nb_token(board, opponent[0]+'_O') == 0:
            print("Aucun pion restant, match nul")
            print(model.str_board_colored(board))
            sys.exit(0)
        else:
            # Swap the players
            active_player, opponent = opponent, active_player

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='python game.py',
                    formatter_class=argparse.RawDescriptionHelpFormatter,
                    description=textwrap.dedent('''\
                            OXOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXO
                            X               Oxono               X
                            O   Daniel Calvez & Vincent Ducot   O
                            X   M1 Bio-Info - 2025              X
                            OXOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXO'''))

    parser.add_argument('player1', help='Nom du 1er joueur (ou IA)')
    parser.add_argument('player2', help='Nom du 2e joueur. '
        'Si le second joueur est une IA, elle sera nommée automatiquement.')

    parser.add_argument('--player1-ia', action='store_true', default=False,
                        help='Si le 1er joueur est une IA.')
    parser.add_argument('--player2-ia', action='store_true', default=False,
                        help='Si le 2e joueur est une IA.')

    parser.add_argument('--ia1-level', default=0,
                        choices = [0,1,2], type=int,
                        help='Niveau de la 1ère IA : 0 (random), 1 (random++), 2 (DQN)')
    parser.add_argument('--ia2-level', default=0,
                        choices = [0,1,2], type=int,
                        help='Niveau de la 2e IA : 0 (random), 1 (random++), 2 (DQN)')
    args = parser.parse_args()
    main(args)
