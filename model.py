import random as rd

EMPTY_CELL = '   '

class TotemException(Exception):
    '''
    Exception related to the token
    '''
    pass

def init_board() -> list[list[str]]:
    '''
    Initialize the 6x6 board, with empty cells
    Place the two totems randomly at the center
    Args
        No args
    Returns
        A 6x6 matrix
    Raise
        No exception
    '''
    board = [[EMPTY_CELL for _ in range(6)] for _ in range(6)]
    pos_totem_O = (rd.randrange(2,4,1), rd.randrange(2,4,1))
    pos_totem_X = (5-pos_totem_O[0],5-pos_totem_O[1])
    board[pos_totem_O[0]][pos_totem_O[1]] = "T_O"
    board[pos_totem_X[0]][pos_totem_X[1]] = "T_X"
    return board

def str_board(board: list[list[str]]) -> str:
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

def find_totem(board: list[list[str]], totem: str) -> tuple[int,int]:
    '''
    Find the position of a totem
    Args
        The board as a matrix
        The totem's name T_X or T_O
    Returns
        The coordinates of the totem, as a tuple (x,y)
    Raise
        TotemException if 
            the totem's name is not T_X or T_O
            the totem has not been found
    '''
    if not (totem == "T_X" or totem == "T_O"):
        raise ValueError("Totem's name is not T_X or T_O.")
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == totem:
                return (i,j)
    raise ValueError("Totem is not in the board.")


def is_landlocked(board: list[list[str]], coord: tuple[int,int]) -> bool:
    '''
    Return if the cell is landlocked, ie surrounded by other pawns, totem or border
    If a totem is landlocked, the player can play its pawn wherever he wants
    Args
        The board as a matrix
        The coordinates of the cell to check
    Returns
        True is the cell is landlocked, else False
    Raise
        ValueError if the cell does not belong to the board
    '''
    x = coord[0]
    y = coord[1]

    # Up
    if x - 1 >= 0:
        if board[x-1][y] == EMPTY_CELL:
            return False
    # Down
    if x + 1 < len(board) - 1:
        if board[x+1][y] == EMPTY_CELL:
            return False
    # Left
    if y - 1 >= 0:
        if board[x][y-1] == EMPTY_CELL:
            return False
    # Right
    if y + 1 < len(board[x]) + 1:
        if board[x][y+1] == EMPTY_CELL:
            return False
        
    return True

def all_free_cells(board: list[list[str]]) -> set[tuple[int,int]]:
    '''
    Return a set of all free cells in the board, ie with no pawn or totem
    Args
        The board as a matrix
    Returns 
        A set of all free cells
    Raise
        No exception
    '''
    cells = [board[row][col] for row in range(len(board)) for col in range(len(board[row])) if board[row][col] == EMPTY_CELL]
    return set(cells)

def row_totem_moves(board: list[list[str]], coord: tuple[int,int], direction: str) -> set[tuple[int,int]]:
    '''
    Compute all totem's possible moves on a line, from the right or left of its position
    Assume that the player can move the totem (ie have enough piece of that symbol)
    Args
        The board as a matrix
        The coordinates of the totem's cell
        The direction "right" or "left" of the totem's moves
    Returns
        A set of possible coordinates
    Raise
        ValueError if direction is neither "right" nor "left"
    '''
    x = coord[0]
    y = coord[1]
    right_border = len(board[x])
    moves = set()

    # List of cells between the totem and the border
    if direction == 'right':
        row_range = range(y+1, right_border)
    elif direction == 'left':
        row_range = range(y-1, 0, -1)
    else:
        raise ValueError("Direction must be 'right' or 'left'.")

    for col in row_range:
        # Add moves until meet an obstacle
        if board[x][col] != EMPTY_CELL:
            break
        else:
            moves.add((x, col))

    # If no move is possible, maybe jumping is possible
    if len(moves) == 0:
        # Find next empty cell
        # @TODO: combine with the previous loop ?
        for col in row_range:
            if board[x][col] == EMPTY_CELL:
                moves.add((x, col))
                break
    return moves

def col_totem_moves(board: list[list[str]], coord: tuple[int,int], direction: str) -> set[tuple[int,int]]:
    '''
    Compute all totem's possible moves on a column, from the up or down of its position
    Assume that the player can move the totem (ie have enough piece of that symbol)
    Args
        The board as a matrix
        The coordinates of the totem's cell
        The direction "up" or "down" of the totem's moves
    Returns
        A set of possible coordinates
    Raise
        ValueError if direction is neither "up" nor "down"
    '''
    x = coord[0]
    y = coord[1]
    down_border = len(board)
    moves = set()

    # List of cells between the totem and the border
    if direction == 'up':
        col_range = range(x-1, 0, -1)
    elif direction == 'down':
        col_range = range(x+1, down_border)
    else:
        raise ValueError("Direction must be 'up' or 'down'.")

    for col in col_range:
        # Add moves until meet an obstacle
        if board[x][col] != EMPTY_CELL:
            break
        else:
            moves.add((x, col))

    # If no move is possible, maybe jumping is possible
    if len(moves) == 0:
        # Find next empty cell
        # @TODO: combine with the previous loop ?
        for col in col_range:
            if board[x][col] == EMPTY_CELL:
                moves.add((x, col))
                break
    return moves

def all_totem_moves(board: list[list[str]], totem: str) -> set[tuple[int,int]]:
    '''
    Compute all totem's possible moves, depending of its position
    Assume that the player can move the totem (ie have enough piece of that symbol)
    Args
        The board as a matrix
        The totem's name T_X or T_O
    Returns
        A set of possible coordinates
    Raise
        TotemException if the totem is not on the board
    '''
    totem_x, totem_y = find_totem(board,  totem)

    all_moves = set()
    
    # If totem's row and column is full, it can move everywhere on the board
    if is_landlocked(board, (totem_x, totem_y)):
        return all_free_cells(board)
    
    # Add all moves in each direction, including cell's jump
    all_moves.union(row_totem_moves(board, (totem_x, totem_y), 'right'))
    all_moves.union(row_totem_moves(board, (totem_x, totem_y), 'left'))
    all_moves.union(col_totem_moves(board, (totem_x, totem_y), 'up'))
    all_moves.union(col_totem_moves(board, (totem_x, totem_y), 'down'))

    print(f"There are {len(all_moves)} moves possibles.")
    return all_moves