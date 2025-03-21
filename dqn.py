'''
Oxono game's IA based on Deep-Q-Learning network
M1 Bio-info project
Daniel Calvez & Vincent Ducot
2025
'''

from torch import flatten, nn, from_numpy, exp
import torch.nn.functional as F
import random
import model 
import numpy as np
from icecream import ic
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

def print_tensor(tensor: np.array):
    '''
    Loop over the layers to print the tensor as a human could understand it
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
        for j, jcol in enumerate(irow):
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
                ic(active_player)
                ic(player1)
                tensor[i,j, 6] = 1

    return tensor

def load_CNN(filepath):
    ''' Load the network from a file '''
    path = Path(filepath)
    # Create a new network if none exists
    if not path.exists():
        return XonoxNetwork()
    model = torch.load(filepath, weights_only=False)
    return model

def write_CNN(model, filepath):
    ''' Save the network into a file '''
    torch.save(model.state_dict(), filepath)

class XonoxNetwork(nn.Module):
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
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return exp(x)

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
    CASES_TOTEM = {0:"A",1:"B",2:"C",3:"D",4:"E",5:"F"}
    move=""
    if output%2 == 0:
        move +="X"
    else:
        move += "O"

    output = output//2
    move += CASES_TOTEM[output%6] + f"{(output//6)%6 +1}"
    output = output//(36)

    move += CASES_TOTEM[output%6] + f"{(output//6)%6 +1}"
    output = output//(36)
    return move

def filter_outputs(result, board, player):
    action_table = [[],[]]
    for i in range(len(result)):
        action = traduce_output(i)
        if model.is_valid_action(board, action, player):
            action_table[0].append(action) 
            action_table[1].append(result[i])
    return action_table

def random_select(action_table):
    action = random.choices(action_table[0], action_table[1], k=1)
    return action

def feedback():
    return

def loss():
    return

cnn_xonox = load_CNN('xonox_network.bbl')

board = [
    ['P_O', '   ', '   ', '   ', '   ', '   '],
    ['   ', '   ', '   ', '   ', '   ', '   '],
    ['   ', 'P_X', '   ', 'P_X', 'T_O', '   '],
    ['   ', 'P_X', 'T_X', 'P_O', 'J_O', '   '],
    ['   ', '   ', 'J_O', '   ', '   ', '   '],
    ['   ', '   ', '   ', '   ', '   ', 'J_X']
]
tensor = convert_board(board, "Paul", "Jeanne", "Paul")
tensor = from_numpy(np.astype(tensor, np.float32))
# Tester si toutes les actions possibles sont prises en compte
""" l = []
for i in range(100000):
    action = traduce_output(i)
    if action in l:
        print(len(l))
        break
    else:
        l.append(action) """

a = XonoxNetwork()

import datetime
start = datetime.datetime.now()
vector = list(a(tensor.unsqueeze(0))[0].tolist())
outp = (filter_outputs(vector, board, "Paul"))
end = datetime.datetime.now()
ic((end-start)/1000)
""" print(outp)
print(random_select(outp)) """