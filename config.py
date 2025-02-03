"""
Useful constants for Oxono game
"""

# Size of the board fixed to 6
SIZE = 6

# unicode characters for drawing the board
FRAME_H_WALL = "─"
FRAME_V_WALL = "│"

FRAME_RIGHT_CROSS = "┤"
FRAME_LEFT_CROSS = "├"
FRAME_MIDDLE_CROSS = "┼"
FRAME_TOP_CROSS = "┬"
FRAME_BOTTOM_CROSS = "┴"

FRAME_TOP_LEFT = "╭"
FRAME_TOP_RIGHT = "╮"
FRAME_BOTTOM_LEFT = "╰"
FRAME_BOTTOM_RIGHT = "╯"

# Exemples d'affichage du plateau, les espaces sont importants.
# En particulier, il n'y en a pas en fin de ligne, même après le F
EXAMPLE1 = """
    A   B   C   D   E   F
  ╭───┬───┬───┬───┬───┬───╮
1 │   │   │   │   │   │   │
  ├───┼───┼───┼───┼───┼───┤
2 │   │   │   │   │   │   │
  ├───┼───┼───┼───┼───┼───┤
3 │   │   │T_O│   │   │   │
  ├───┼───┼───┼───┼───┼───┤
4 │   │   │   │T_X│   │   │
  ├───┼───┼───┼───┼───┼───┤
5 │   │   │   │   │   │   │
  ├───┼───┼───┼───┼───┼───┤
6 │   │   │   │   │   │   │
  ╰───┴───┴───┴───┴───┴───╯
"""
# Autre exemple avec quelques pions posés
EXAMPLE2 = """
    A   B   C   D   E   F
  ╭───┬───┬───┬───┬───┬───╮
1 │   │   │   │   │   │   │
  ├───┼───┼───┼───┼───┼───┤
2 │   │   │   │   │   │   │
  ├───┼───┼───┼───┼───┼───┤
3 │   │T_X│J_X│P_X│T_O│   │
  ├───┼───┼───┼───┼───┼───┤
4 │   │P_X│   │   │J_O│   │
  ├───┼───┼───┼───┼───┼───┤
5 │   │   │   │   │   │   │
  ├───┼───┼───┼───┼───┼───┤
6 │   │   │   │   │   │   │
  ╰───┴───┴───┴───┴───┴───╯
"""

if __name__ == "__main__":
    print(EXAMPLE1)
    print(EXAMPLE2)
