'''
Oxono game's IA based on Deep-Q-Learning network
M1 Bio-info project
Daniel Calvez & Vincent Ducot
2025
'''

from torch import flatten, nn, from_numpy
import torch.nn.functional as F
import random
import model 
import numpy as np
from icecream import ic
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNModel(nn.Module):
    def __init__(self):
        super(DQNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        # ???

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 16 * 5 * 5)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x
    
def print_tensor(tensor):
    for k in range(7):
        print(f"Couche{k}")
        for i in range(6):
            print(" ".join(str(tensor[i,j,k]) for j in range(6)))

def convert_board(board, player1, player2, active_player) -> np.array:
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

def read_CNN(model, filepath):
    model = torch.load(filepath, weights_only=False)
    return model

def write_CNN(model, filepath):
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
        self.fc1 = nn.Linear(864*6*6, 2304)
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
        return x

def traduce_output(prefered_output: int):
    CASES_TOTEM = {0:"A",1:"B",2:"C",3:"D",4:"E",5:"F"}
    move=""
    if prefered_output%2 == 0:
        move +="X"
    else:
        move += "O"

    prefered_output = prefered_output//2
    move += CASES_TOTEM[prefered_output%6] + f"{(prefered_output//6)%6 +1}"
    prefered_output = prefered_output//(36)

    move += CASES_TOTEM[prefered_output%6] + f"{(prefered_output//6)%6 +1}"
    prefered_output = prefered_output//(36)
    return move

def filter_outputs():
    return

def random_select():
    return

def feedback():
    return

def loss():
    return


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
ic(tensor)
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
print(a(tensor.unsqueeze(0)))