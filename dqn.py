'''
Oxono game's IA based on Deep-Q-Learning network
M1 Bio-info project
Daniel Calvez & Vincent Ducot
2025
'''

from torch import flatten
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
import random
import model 
import numpy as np
from icecream import ic

def convert_board(board, player1, player2, active_player) -> np.array:
    tensor = np.zeros((6,6,7), dtype=np.int8)

    for i, irow in enumerate(board):
        for j, jcol in enumerate(irow):
            cell_content = board[i][j].strip()
            # ic(i)
            # ic(j)
            # ic(cell_content)
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
                tensor[i,j, 6] = 2
            else:
                ic(active_player)
                ic(player1)
                tensor[i,j, 6] = 1
    print(tensor[0,0,6])
    ic(tensor[:,:,6])
    ic(tensor)
    return tensor

def read_CNN():
    return

def write_CNN():
    return

class CNN():
    def __init__(self, numChannels, classes):
        super(XonoxNetwork, self).__init__()
        self
def traduce_output():
    return

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
