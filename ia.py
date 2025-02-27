'''
Oxono game's IA
M1 Bio-info project
Daniel Calvez & Vincent Ducot
2025
'''

import random
import model as model

def name():
    '''
    Returns the IA's name
    Args
        None
    Returns
        The name as a string
    Exception
        No exception
    '''
    return "DeepTic"

def ask_play(board: list[list[str]], player: str, opponent: str) -> str:
    '''
    Ask the IA its action
    Args
        The board as a matrix
        The player's name
        The opponent's name
    Returns
        The action as a string
    Exception
        No exception
    '''
    totems = ['T_O', 'T_X']
    totem = random.choice(totems)
    totem_coords = model.find_totem(board, totem)
    totem_move = random.choice(list(model.all_totem_moves(board, totem)))
    pawn_move = random.choice(list(model.all_token_drops(board, totem_coords)))
    action = f"{totem[-1]}{model.reverse_convert_coord(totem_move)}{model.reverse_convert_coord(pawn_move)}"
    print(f"Action computed {action}")
    return action