'''
Oxono game's model
M1 Bio-info project
Daniel Calvez & Vincent Ducot
2025
'''

import random as rd
from icecream import ic
from colorama import Fore, Style

EMPTY_CELL = '   '
PINK_PLAYER = ""
ic.configureOutput(includeContext=True)

class TotemException(Exception):
    '''
    Exception related to the token
    '''

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

def set_player1(player: str) -> None:
    '''
    Defines the player one, which have the pink pawn's color
    ! Modify the global variable !
    Args
        The player1's name
    Returns
        None
    Exception
        No exception
    '''
    global PINK_PLAYER
    PINK_PLAYER = player

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
    displayed_board = "\n    A   B   C   D   E   F\n  ╭───┬───┬───┬───┬───┬───╮\n"
    for i in range(len(board)):
        displayed_board += f"{i+1} "+''.join([str(f"|{cell}") for cell in board[i]])+"|\n"
        if i < 5:
            displayed_board += "  ├───┼───┼───┼───┼───┼───┤\n"
    displayed_board += "  ╰───┴───┴───┴───┴───┴───╯"
    return displayed_board

def str_board_colored(board: list[list[str]], player1: str, player2: str) -> str:
    '''
    Returns a string representing the board, with color for each player and totems
    Args
        The board as a matrix
        The first player's name
        The second player's name
    Returns
        A string representing the board
    Raise
        No exception
    '''
    displayed_board = "\n    A   B   C   D   E   F\n  ╭───┬───┬───┬───┬───┬───╮\n"
    for i, line in enumerate(board):
        displayed_board += f"{i+1} "
        for cell in line:
            # Colorize according the cell type
            if cell[0] == 'T':
                color_cell = Fore.CYAN + cell + Style.RESET_ALL
            elif cell[0] == PINK_PLAYER[0]:
                color_cell = Fore.MAGENTA + cell + Style.RESET_ALL
            else:
                color_cell = Fore.GREEN + cell + Style.RESET_ALL
            displayed_board += f"|{color_cell}"
        displayed_board += "|\n"
        if i < 5:
            displayed_board += "  ├───┼───┼───┼───┼───┼───┤\n"
    displayed_board += "  ╰───┴───┴───┴───┴───┴───╯"
    return displayed_board

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
    if not totem in ("T_X", "T_O"):
        raise ValueError("Totem's name is not T_X or T_O.")
    for i, line in enumerate(board):
        for j, _ in enumerate(line):
            if board[i][j] == totem:
                return (i,j)
    raise ValueError("Totem is not in the board.")

def nb_token(board: list[list[str]], token: str) -> int:
    '''
    Check how many of the token given in parameter remains
    Args
        The board a a matrix
        The token as a string
    Returns
        Returns the number of remaining tokens as an integer
    Raise
        No exception
    '''
    total_count = 8
    for line in board:
        for elem in line:
            if elem == token:
                total_count -= 1
    return total_count

def is_landlocked(board: list[list[str]], coord: tuple[int,int]) -> bool:
    '''
    Return if the cell is landlocked, ie surrounded by other pawns, totem or border
    If a totem is landlocked, the player can make it jump on its row or column
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
    if y + 1 < len(board[x]) - 1:
        if board[x][y+1] == EMPTY_CELL:
            return False

    return True

def is_totem_fully_landlocked(board: list[list[str]], coord: tuple[int,int]) -> bool:
    '''
    Return if the totem is fully landlocked, ie its row or column are full
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
        row = x-1
        while row >= 0:
            if board[row][y] == EMPTY_CELL:
                return False
            row -= 1

    # Down
    if x + 1 < len(board) - 1:
        for row in range(x+1, len(board)):
            if board[row][y] == EMPTY_CELL:
                return False
            
    # Left
    if y - 1 >= 0:
        col = y-1
        while col >= 0:
            if board[x][col] == EMPTY_CELL:
                return False
            col -= 1

    # Right
    if y + 1 < len(board[x]) - 1:
        for col in range(y+1, len(board[x])):
            if board[x][col] == EMPTY_CELL:
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
    cells = [board[row][col] for row in range(len(board)) \
             for col in range(len(board[row])) if board[row][col] == EMPTY_CELL]
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
        moves.add((x, row))

    # If no move is possible, maybe jumping is possible
    if len(moves) == 0 and is_landlocked(board, coord):
        # Find next empty cell
        # @TODO: combine with the previous loop ?
        #print(f"Testing jump for x = {x}")
        #print(f"Row range : {list(row_range)}")
        for row in row_range:
            #print(f"Empty cell in ({x},{row}) ? : {board[x][row]}")
            if board[x][row] == EMPTY_CELL:
                #print("yes")
                moves.add((x, row))
                break
    #print(f"Found {len(moves)} for direction {direction} : {moves}")
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
        moves.add((col, y))

    # If no move is possible, maybe jumping is possible
    if len(moves) == 0 and is_landlocked(board, coord):
        #print(f"Testing jump for y = {y}")
        #print(f"Row range : {list(col_range)}")
        # Find next empty cell
        # @TODO: combine with the previous loop ?
        for col in col_range:
            #print(f"Empty cell in ({col},{y}) ? : {board[col][y]}")
            if board[col][y] == EMPTY_CELL:
                moves.add((col, y))
                break
    #print(f"Found {len(moves)} for direction {direction} : {moves}")
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
    if is_totem_fully_landlocked(board, (totem_x, totem_y)):
        return all_free_cells(board)

    # Add all moves in each direction, including cell's jump
    all_moves.update(row_totem_moves(board, (totem_x, totem_y), 'right'))
    all_moves.update(row_totem_moves(board, (totem_x, totem_y), 'left'))
    all_moves.update(col_totem_moves(board, (totem_x, totem_y), 'up'))
    all_moves.update(col_totem_moves(board, (totem_x, totem_y), 'down'))

    #print(f"There are {len(all_moves)} moves possibles : {all_moves}")
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

    ic(totem_coord)
    accessible_positions=[]
    if(totem_coord[0]-1 >= 0 and board[totem_coord[0]-1][totem_coord[1]] == EMPTY_CELL):
        accessible_positions.append((totem_coord[0]-1, totem_coord[1]))
    if(totem_coord[0]+1 < len(board) and board[totem_coord[0]+1][totem_coord[1]] == EMPTY_CELL):
        accessible_positions.append((totem_coord[0]+1, totem_coord[1]))
    if(totem_coord[1]-1 >= 0 and board[totem_coord[0]][totem_coord[1]-1] == EMPTY_CELL):
        accessible_positions.append((totem_coord[0], totem_coord[1]-1))
    if(totem_coord[1]+1 < len(board[0]) and board[totem_coord[0]][totem_coord[1]+1] == EMPTY_CELL):
        accessible_positions.append((totem_coord[0], totem_coord[1]+1))
    return set(accessible_positions)

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

    return len(player1)>0 and len(player2)>0 and (player1[0]!=player2[0]) and \
        (player1[0]!="T") and (player2[0]!="T") and \
            (player1[0].isupper()) and (player2[0].isupper())

def convert_coord(coord: str) -> tuple[int,int]:
    '''
    Convert the coordinates given by the player as a position in the board
    Examples
        A2 => (1,0)
        C5 => (4,2)
    Args
        The coordinates as a string
    Returns
        The coordinates as a tuple
    '''
    valid_letters = ["A","B","C","D","E","F"]
    # Checks if coordinates format is valid
    if(len(coord) != 2 or (coord[0] not in valid_letters) or
        (not coord[1].isnumeric) or (not 1 <= int(coord[1]) <= 6)):
        raise ValueError(f"{coord} n'est pas correcte")

    return (int(coord[1]) - 1, valid_letters.index(coord[0]))

def reverse_convert_coord(coord: tuple[int, int]) -> str:
    '''
    Convert a tuple of coordinates into a string position on the board
    Examples
        (1, 0) => "A2"
        (4, 2) => "C5"
    Args:
        The coordinates as a tuple
    Returns:
        The coordinates as a string
    '''
    valid_letters = ["A","B","C","D","E","F"]
    
    # Check if the tuple coordinates are within valid bounds
    if not (0 <= coord[0] <= 5 and 0 <= coord[1] <= 5):
        raise ValueError(f"Coordonnées {coord} incorrectes")
    
    # Return the coordinates in string format
    return f"{valid_letters[coord[1]]}{coord[0] + 1}"

def is_action(action: str) -> bool:
    """
    Checks is the action format is correct
    Args
        The action as a string
    Returns
        The correctness of the action as a boolean
    """
    return (len(action) == 5) and (action[0]=="O" or action[0]=="X") and \
        action[1].isalpha() and 'A' <= action[1] <= 'F' and \
            action[3].isalpha() and 'A' <= action[3] <= 'F' and \
            action[2].isnumeric() and 1 <= int(action[2]) <= 6 and \
                action[4].isnumeric() and 1 <= int(action[4]) <= 6

def is_valid_action(board: list[list[str]], action: str, player: str ) -> bool :
    '''
    Look if the action is valid and doable
    Args
        The board as a matrix
        The action as a string
        The player's name
    Returns
        The validity of the action as a boolean
    Exception
        No exception
    '''
    if not is_action(action):
        print("Not an action")
        return False
    if action[0] == "X":
        if nb_token(board, player[0]+"_X") <= 0:
            print("No token left")
            return False
        totem = "T_X"
    else:
        if nb_token(board, player[0]+"_O") <= 0:
            print("No token left")
            return False
        totem = "T_O"

    totem_moves = all_totem_moves(board, totem)
    totem_coord = convert_coord(action[1:3])
    curr_coord_totem = find_totem(board,totem)
    if totem_coord not in totem_moves:
        print("Totem move impossible")
        return False

    token_drops = all_token_drops(board, totem_coord)
    token_coord = convert_coord(action[3:5])

    if is_landlocked(board, totem_coord):
        token_drops = all_free_cells(board)
        token_drops.remove(totem_coord)
        token_drops.append(curr_coord_totem)

    if abs(curr_coord_totem[0] - totem_coord[0] + curr_coord_totem[1] - totem_coord[1]) == 1:
        token_drops.add(curr_coord_totem)

    if token_coord not in token_drops:
        print("Token move impossible")
        return False
    return True

def ask_play(board: list[list[str]], player: str, opponent: str) -> str:
    '''
    Ask the player its action
    Args
        The board as a matrix
        The player's name
        The opponent's name
    Returns
        The action as a string
    Exception
        No exception
    '''

    print(str_board_colored(board, player, opponent))
    valid_action = False
    action = ""
    while not valid_action:
        action = input(f"Vous êtes {player}, quelle action souhaitez-vous faire ? ")
        valid_action = is_valid_action(board, action, player)
        print(f"Is {action} a valid action ? {valid_action}")

    return action

def is_winner(board: list[list[str]], player: str, coord: tuple[int,int]) -> bool:
    '''
    Look if a move from a player is a winning one
    Args
        The board as a matrix
        The player's name
        The coordinates of the move
    Returns
        The player's victory as a boolean
    Exception
        No exception
    '''
    # Get the token from the coordinates and check if it belongs to the player
    token = board[coord[0]][coord[1]]
    if token[0] != player[0]:
        return False

    symbol_score = 0
    color_score = 0
    vertical_pos = coord[0]
    offset = -1
    is_color = True
    is_symbol = True
    # This loops parse backward vertically, checking the symbol and color
    # and going the other way if there is no token, a different symbol and color or if out of bounds
    while vertical_pos<len(board):
        if(vertical_pos<0 or (not(is_color or is_symbol) and offset <0)):
            vertical_pos=coord[0] +1
            offset = 1
            is_color = True
            is_symbol = True
        if(is_symbol and board[vertical_pos][coord[1]][2]== token[2] and token[0]!="T"):
            symbol_score +=1
        else:
            is_symbol = False

        if(is_color and board[vertical_pos][coord[1]][0] == token[0]):
            color_score += 1
        else:
            is_color=False
        if(symbol_score==4 or color_score==4):
            return True
        if(not(is_color or is_symbol)and offset>0):
            break
        vertical_pos += offset

    symbol_score = 0
    color_score = 0
    horizontal_pos = coord[1]
    offset = -1
    is_color = True
    is_symbol = True
    # Same loop but horizontal
    while horizontal_pos<len(board):
        if(horizontal_pos<0 or  (not(is_color or is_symbol) and offset <0)):
            horizontal_pos=coord[1] +1
            offset = 1
            is_color = True
            is_symbol = True
        if(is_symbol and board[coord[0]][horizontal_pos][2]==token[2] and token[0]!="T"):
            symbol_score +=1
        else:
            is_symbol = False

        if(is_color and board[coord[0]][horizontal_pos][0] == token[0]):
            color_score += 1
        else:
            is_color=False
        if(symbol_score==4 or color_score==4):
            return True
        if(not(is_color or is_symbol)and offset>0):
            break
        horizontal_pos += offset
    return False
