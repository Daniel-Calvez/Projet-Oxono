'''
Oxono game's IA
M1 Bio-info project
Daniel Calvez & Vincent Ducot
2025
'''

import random
import model as model
from icecream import ic
import dqn

def name() -> str:
    '''
    Returns the IA's name
    Args
        None
    Returns
        The name as a string
    Exception
        No exception
    '''
    return "XonoX_sous_bbl"

def all_pawn_moves(board: list[list[str]], totem_move: tuple[int,int], totem: str) -> set[tuple[int,int]]:
    '''
    Compute all possibilities for placing a pawn
    Since the intelligence is also in is_valid_action function, we can't only call all_token_drops function
        Args
        The board as a matrix
        The coordinates of the totem's move as a tuple
        The totem as a string
    Returns
        A list of possible moves as a set of tuple
    Exception
        No exception
    '''
    curr_coord_totem = model.find_totem(board, totem)

    # First get all evident positions
    pawn_moves = model.all_token_drops(board, totem_move)
    # Then add special cases
    if model.is_landlocked(board, totem_move):
        pawn_moves = model.all_free_cells(board)
        #ic(token_drops)
        pawn_moves.discard(totem_move)
        pawn_moves.add(curr_coord_totem)

    elif abs(curr_coord_totem[0] - totem_move[0] + curr_coord_totem[1] - totem_move[1]) == 1:
        pawn_moves.add(curr_coord_totem)

    return pawn_moves

def try_to_win(board: list[list[str]], totem: str, player: str) -> str:
    '''
    Check if there is a winning action with this totem
    Args
        The board as a matrix
        The totem as a string
        The player's name
    Return 
        The action as a string, or None if there is no winning action
    Exception
        No exception
    '''

    all_totem_moves = model.all_totem_moves(board, totem)
    for move in all_totem_moves:
        pawn_moves = all_pawn_moves(board, move, totem)
        for drop in pawn_moves:
            print(f"Test is {drop} is winning")
            if model.is_winner(board, player, drop):
                action = f"{totem[-1]}{model.reverse_convert_coord(move)}{model.reverse_convert_coord(drop)}"
                print(f"Go win with {action}")
                return action
    return None

def random_play(board: list[list[str]], ia_level: int, player: str) -> str:
    ''' Compute a random action
    Args
        The board as a matrix
        The level : 
            if 0, plays random
            if 1, first tries to win the game, if not possible plays random
        The player's name
    Return 
        The action as a string
    Exception
        No exception
    '''

    totems = ['T_O', 'T_X']
    # Try to win with each totem
    if ia_level == 1:
        action = try_to_win(board, 'T_O', player)
        if action is None:
            action = try_to_win(board, 'T_X', player)
        if action is not None:
            return action
        
    # Else, play random
    totem = random.choice(totems)
    totem_coords = model.find_totem(board, totem)
    all_totem = model.all_totem_moves(board, totem)
    #ic(all_totem)
    totem_move = random.choice(list(all_totem))
    #ic(totem_move)
    all_drops = all_pawn_moves(board, totem_move, totem)
    #ic(all_drops)
    pawn_move = random.choice(list(all_drops))
    action = f"{totem[-1]}{model.reverse_convert_coord(totem_move)}{model.reverse_convert_coord(pawn_move)}"
    return action

def ask_play(board: list[list[str]], player: str, opponent: str, ia_level: int = 0) -> str:
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
    if ia_level == 0 or ia_level == 1:
        action = random_play(board, ia_level, player)
    elif ia_level == 2:
        action = dqn_play(board, player, cnn)

    print(f"Action computed {action}")
    return action
