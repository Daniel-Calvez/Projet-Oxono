import random as rd
from icecream import ic

EMPTY_CELL = '   '
ic.configureOutput(includeContext=True)

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

def nb_token(board: list[list[str]], token: str) -> int:
    total_count = 8
    for line in board:
        for elem in line:
            if elem == token: total_count -= 1 
    return total_count


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
        row_range = range(y-1, -1, -1)
    else:
        raise ValueError("Direction must be 'right' or 'left'.")

    for row in row_range:
        # Add moves until meet an obstacle
        if board[x][row] != EMPTY_CELL:
            break
        else:
            moves.add((x, row))

    # If no move is possible, maybe jumping is possible
    if len(moves) == 0:
        # Find next empty cell
        # @TODO: combine with the previous loop ?
        for row in row_range:
            if board[x][row] == EMPTY_CELL:
                moves.add((x, row))
                break
    print(f"Found {len(moves)} for direction {direction}")
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
        col_range = range(x-1, -1, -1)
    elif direction == 'down':
        col_range = range(x+1, down_border)
    else:
        raise ValueError("Direction must be 'up' or 'down'.")

    for col in col_range:
        # Add moves until meet an obstacle
        if board[col][y] != EMPTY_CELL:
            break
        else:
            moves.add((col, y))

    # If no move is possible, maybe jumping is possible
    if len(moves) == 0:
        # Find next empty cell
        # @TODO: combine with the previous loop ?
        for col in col_range:
            if board[col][y] == EMPTY_CELL:
                moves.add((col, y))
                break
    print(f"Found {len(moves)} for direction {direction}")
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
    all_moves.update(row_totem_moves(board, (totem_x, totem_y), 'right'))
    all_moves.update(row_totem_moves(board, (totem_x, totem_y), 'left'))
    all_moves.update(col_totem_moves(board, (totem_x, totem_y), 'up'))
    all_moves.update(col_totem_moves(board, (totem_x, totem_y), 'down'))

    print(f"There are {len(all_moves)} moves possibles.")
    return all_moves

def move_totem(board: list[list[str]], totem: str, coord: tuple[int, int]) -> None:
    '''
    Move the totem on the board, update the board
    Assume the movement is possible
    Args
        The board as a matrix
        The name of the totem to move
        The new coordinates of the totem to move
    Returns
        Nothing
    Exception
        No exception
    '''
    actual_coord = find_totem(board, totem)
    board[actual_coord[0]][actual_coord[1]] = EMPTY_CELL
    board[coord[0]][coord[1]] = totem
    
def all_token_drops(board: list[list[str]], totem_coord: tuple[int,int]) -> set[tuple[int,int]]:
    """
    Compute every move the player can do after moving the totem
    Args
        The board as a matrix
        The coordinates of the totem as a tuple
    Returns
        A list of possible moves as a set of tuple
    """

    accessible_positions=[]
    if(totem_coord[0] - 1 >= 0 and board[totem_coord[0]-1, totem_coord[1]] != EMPTY_CELL): accessible_positions.append((totem_coord[0]-1, totem_coord[1]))
    if(totem_coord[0] + 1 < len(board) and board[totem_coord[0]+1, totem_coord[1]] != EMPTY_CELL): accessible_positions.append((totem_coord[0]+1, totem_coord[1]))
    if(totem_coord[1] - 1 >= 0 and board[totem_coord[0], totem_coord[1]-1] != EMPTY_CELL): accessible_positions.append((totem_coord[1], totem_coord[1]-1))
    if(totem_coord[1] + 1 < len(board[0]) and board[totem_coord[0], totem_coord[1]+1] != EMPTY_CELL): accessible_positions.append((totem_coord[1], totem_coord[1]+1))
    return accessible_positions

def valid_player_names(player1: str, player2:str) -> bool:
    """
    Check if player names are valid, starting with a capital letter
    Starting with a different character and doesn't start with the letter T
    Args
        The first player name as a string
        The second player name as a string
    Returns
        The validity of the player names as a boolean
    """

    return len(player1)>0 and len(player2)>0 and (player1[0]!=player2[0]) and (player1[0]!="T") and (player2[0]!="T") and (player1[0].isupper()) and (player2[0].isupper())

def convert_coord(coord: str) -> tuple[int,int]:
    """
    Convert the coordinates given by the player as a position in the board
    Args
        The coordinates as a string
    Returns
        The coordinates as a tuple
    """
    valid_letters = ["A","B","C","D","E","F"]
    # Checks if coordinates format is valid
    if(1 > len(coord) > 3 and (coord[0] not in valid_letters) and (not coord[1].isnumeric) and (1 > int(coord[1]) > 6)):
        raise ValueError(f"{coord} n'est pas correcte")
    
    return (valid_letters.index(coord[0]), int(coord[1]) - 1)

def is_action(action: str) -> bool:
    """
    Checks is the action format is correct
    Args
        The action as a string
    Returns
        The correctness of the action as a boolean
    """
    return (5==len(action)) and action[0].isalpha() and action[1].isalpha() and action[3].isalpha() and action[2].isnumeric and action[4].isnumeric

def ask_play(board: list[list[str]], player: str, opponent: str) -> str:
    '''
    Ask the player its action
    The action should be valid (for example XB1C2)
    Args
        The board as a matrix
        The player's name
        The opponent's name
    Returns
        The action as a string
    Exception
        No exception
    '''
    
    print(str_board(board))
    valid_action = False
    action = ""
    while not valid_action:
        input(f"Vous êtes {player}, quelle action souhaitez-vous faire ? ", action)
        valid_action = is_valid_action(action)

    return action