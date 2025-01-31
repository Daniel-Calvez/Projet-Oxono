import random as rd

def init_board() -> list[list[str]]:
    '''
    Initialize the 6x6 board, with empty cases
    Place the two totems randomly at the center
    Args
        No args
    Returns
        A 6x6 matrix
    Raise
        No exception
    '''
    board = [["   " for _ in range(6)] for _ in range(6)]
    pos_totem_O = (rd.randrange(2,4,1), rd.randrange(2,4,1))
    pos_totem_X = (5-pos_totem_O[0],5-pos_totem_O[1])
    board[pos_totem_O[0]][pos_totem_O[1]] = "T_O"
    board[pos_totem_X[0]][pos_totem_X[1]] = "T_X"
    return board

def str_board(board:list[list[str]]) -> str:
    '''
    Returns a string representing the board
    Args
        The board as a matrix
    Returns
        A string representing the board
    Raise
        No exception
    '''
    return '\n'.join(['\t'.join([str(f"'{cell}'") for cell in row]) for row in board])