import random as rd

def init_board() -> list[list[str]]:
    board = [["   " for _ in range(6)] for _ in range(6)]
    pos_totem_O = (rd.randrange(2,4,1), rd.randrange(2,4,1))
    pos_totem_X = (5-pos_totem_O[0],5-pos_totem_O[1])
    board[pos_totem_O[0],pos_totem_O[1]] = "T_O"
    board[pos_totem_X[0],pos_totem_X[1]] = "T_X"

