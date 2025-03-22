'''
Oxono game's IA based on Deep-Q-Learning network
M1 Bio-info project
Daniel Calvez & Vincent Ducot
2025
'''

from pathlib import Path
from torch import nn, load, save, exp, from_numpy
import torch.nn.functional as F
import numpy as np
# from icecream import ic
import model

class XonoxNetwork(nn.Module):
    '''
    This class represents the CNN model
    '''
    def __init__(self):
        super(XonoxNetwork, self).__init__()
        self.conv1 = nn.Conv2d(6, 216, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(216)
        self.conv2 = nn.Conv2d(216, 432, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(432)
        self.conv3 = nn.Conv2d(432, 864, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(864)
        self.fc1 = nn.Linear(36288, 2304)
        self.fc2 = nn.Linear(2304, 1152)
        self.fc3 = nn.Linear(1152, 2592)

    def forward(self, x):
        '''
        Process to pass data between layers
        '''
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return exp(x)

def print_tensor(tensor: np.array):
    '''
    Loop over the layers to print the tensor as a human could understand it
    Args
        The board as a numpry array tensor
    Returns
        Nothing
    Exception
        No exception
    '''
    for k in range(7):
        print(f"Couche{k}")
        for i in range(6):
            print(" ".join(str(tensor[i,j,k]) for j in range(6)))

def convert_board(board: list[list[str]], player1: str, player2: str, active_player: str) -> np.array:
    '''
    Convert the board into a numpy tensor
    7 layers : player1 X, player1 O, player2 X, player2 O, totem X, totem O, active player
    Args:
        the board as a matrix
        player 1 name
        player 2 name
        active player name
    Returns
        the board as a numpy tensor
    Exception
        No exception
    '''
    tensor = np.zeros((6,6,7), dtype=np.int8)

    for i, irow in enumerate(board):
        for j, _ in enumerate(irow):
            cell_content = board[i][j].strip()
            prefix_p1 = player1[0]+'_'
            prefix_p2 = player2[0]+'_'
            if cell_content == prefix_p1+"X":  # X du joueur 1
                tensor[i,j, 0] = 1
            elif cell_content == prefix_p1+"O":  # O du joueur 1
                tensor[i,j, 1] = 1
            elif cell_content == prefix_p2+"X":  # X du joueur 2
                tensor[i,j, 2] = 1
            elif cell_content == prefix_p2+"O":  # O du joueur 2
                tensor[i,j, 3] = 1
            elif cell_content == "T_X":  # Totem X
                tensor[i,j, 4] = 1
            elif cell_content == "T_O":  # Totem O
                tensor[i,j, 5] = 1

            if active_player == player1:
                tensor[i,j, 6] = 0
            else:
                tensor[i,j, 6] = 1

    return tensor

def load_cnn(filepath: str) -> XonoxNetwork:
    '''
    Load the network from a file.
    If the file does not exist, create a new model.
    The file is a state dict format, serialized with pickle
    https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
    Args
        The file's path as a string
    Return 
        An object XonoxNetwork representing the CNN
    Exception
        No exception
    '''
    path = Path(filepath)
    # Create a new network if none exists
    xonox_model = XonoxNetwork()
    if path.exists():
        xonox_model.load_state_dict(load(filepath, weights_only=True))
    return xonox_model

def write_cnn(xonox_model: XonoxNetwork, filepath: str):
    '''
    Save the network into a file
    Args
        The CNN model
        The file's path as a string
    Returns
        Nothing
    Exception
        No exception
    '''
    save(xonox_model.state_dict(), filepath)

def traduce_output(output: int) -> str:
    '''
    Convert the output of the CNN to an Oxono action
    Args:
        output : index of the CNN's output
    Returns
        action in Oxono format
    Exception:
        No Exception
    '''
    CASES_TOTEM = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F"}
    move = ""
    if output % 2 == 0:
        move +="X"
    else:
        move += "O"

    output = output // 2
    move += CASES_TOTEM[output % 6] + f"{(output // 6) % 6 +1}"
    output = output // (36)

    move += CASES_TOTEM[output % 6] + f"{(output // 6) % 6 +1}"
    output = output // (36)
    return move

def filter_outputs(result: list[int], board: list[list[str]], player: str):
    '''
    Filter the CNN's output to keep only allowed moves
    Args
        CNN's output as a list
        The board as a matrix
        The player's name
    Return
        A list with allowed moves and the associated score
    Exception
        No exception
    '''
    action_table = [[],[]]
    for i, res in enumerate(result):
        action = traduce_output(i)
        if model.is_valid_action(board, action, player):
            action_table[0].append(action)
            action_table[1].append(res)
    return action_table

def random_select(action_table) -> str:
    '''
    Pick an action within the possibles one.
    The choice is weighted by the score given by the CNN
    Args
        A list of allowed moves and the associated score
    Returns
        An action as a string
    Exception
        No exception
    '''
    max_pos = action_table[1].index(max(action_table[1]))
    action = action_table[0][max_pos]
    return action

def dqn_play(cnn: list[int], board: list[list[str]], player1: str, player2: str, active_player: str) -> str:
    '''
    Compute the best move according to the DQN network
    Args
        The CNN output as a list
        The board as a matrix
        Player1's name
        Player2's name
        Current player name
    Returns
        An action as a string
    Exception
        No exception
    '''
    tensor = convert_board(board, player1, player2, active_player)
    tensor = from_numpy(np.astype(tensor, np.float32))
    outputs = filter_outputs(cnn(tensor.unsqueeze(0))[0].tolist(), board, active_player)
    return random_select(outputs)

# Load or create a CNN model when this module is loaded
cnn_xonox = load_cnn('xonox_network.bbl')


""" board = [
    ['P_O', '   ', '   ', '   ', '   ', '   '],
    ['   ', '   ', '   ', '   ', '   ', '   '],
    ['   ', 'P_X', '   ', 'P_X', 'T_O', '   '],
    ['   ', 'P_X', 'T_X', 'P_O', 'J_O', '   '],
    ['   ', '   ', 'J_O', '   ', '   ', '   '],
    ['   ', '   ', '   ', '   ', '   ', 'J_X']
]

now = datetime.datetime.now()
action = dqn_play(cnn_xonox, board, "Paul", "Jeanne", "Paul")

 """
