"""
Oxono Unit Tests
"""

from io import StringIO
import pytest
from model import TotemException
oxo = pytest.importorskip("model")

class TestFunctionsExist:
    """Test if each mandatory function exists"""

    def test_valid_player_names(self):
        """Test if valid_player_names function exists"""
        assert hasattr(oxo.valid_player_names, "__call__")

    def test_init_board(self):
        """Test if init_board function exists"""
        assert hasattr(oxo.init_board, "__call__")

    def test_str_board(self):
        """Test if str_board function exists"""
        assert hasattr(oxo.str_board, "__call__")

    def test_convert_coord(self):
        """Test if convert_coord function exists"""
        assert hasattr(oxo.convert_coord, "__call__")

    def test_is_action(self):
        """Test if is_action function exists"""
        assert hasattr(oxo.is_action, "__call__")

    def test_nb_token(self):
        """Test if nb_token function exists"""
        assert hasattr(oxo.nb_token, "__call__")

    def test_find_totem(self):
        """Test if find_totem function exists"""
        assert hasattr(oxo.find_totem, "__call__")

    def test_all_totem_moves(self):
        """Test if all_totem_moves function exists"""
        assert hasattr(oxo.all_totem_moves, "__call__")

    def test_all_token_drops(self):
        """Test if all_token_drops function exists"""
        assert hasattr(oxo.all_token_drops, "__call__")

    def test_move_totem(self):
        """Test if move_totem function exists"""
        assert hasattr(oxo.move_totem, "__call__")

    def test_is_valid_action(self):
        """Test if is_valid_action function exists"""
        assert hasattr(oxo.is_valid_action, "__call__")

    def test_ask_play(self):
        """Test if ask_play function exists"""
        assert hasattr(oxo.ask_play, "__call__")

    def test_is_winner(self):
        """Test if is_winner function exists"""
        assert hasattr(oxo.is_winner, "__call__")

class TestReturnedType:
    """Test the correct typing of returned values"""

    def test_valid_player_names(self):
        """test that function valid_player_names returns a bool"""
        assert isinstance(oxo.valid_player_names("B", "C"), bool)

    def test_init_board(self):
        """test that function init_board returns a list"""
        assert isinstance(oxo.init_board(), list)

    def test_str_board(self):
        """test that function str_board returns a str"""
        assert isinstance(oxo.str_board([["   "] * 6] * 6), str)

    def test_convert_coord(self):
        """test that function convert_coord returns a tuple"""
        assert isinstance(oxo.convert_coord("A2"), tuple)

    def test_is_action(self):
        """test that function is_action returns a bool"""
        assert isinstance(oxo.is_action("XA2C3"), bool)

    def test_nb_token(self):
        """test that function nb_token returns a bool"""
        assert isinstance(oxo.nb_token([["   "] * 6] * 6, "F_X"), int)

    def test_find_totem(self):
        """test that function find_totem returns a tuple"""
        assert isinstance(oxo.find_totem([["T_X"] * 6] * 6, "T_X"), tuple)

    def test_all_totem_moves(self):
        """test that function all_totem_moves returns a set"""
        assert isinstance(oxo.all_totem_moves([["T_X"] * 6] * 6, "T_X"), set)

    def test_all_token_drops(self):
        """test that function all_token_drops returns a set"""
        assert isinstance(oxo.all_token_drops([["T_X"] * 6] * 6, (3, 4)), set)

    def test_move_totem(self):
        """test that function move_totem is None"""
        assert oxo.move_totem([["T_X"] * 6] * 6, "T_X", (3, 4)) is None

    def test_is_valid_action(self):
        """test that function is_valid_action returns a bool"""
        board = [["   " for _ in range(6)] for _ in range(6)]
        board[2][2] = "T_X"
        board[3][3] = "T_O"
        oxo.set_player1("Bobby")
        assert isinstance(oxo.is_valid_action(board, "TC2C1", "Bobby"), bool)
        assert isinstance(oxo.is_valid_action(board, "XC2C1", "Bobby"), bool)

    def test_ask_play(self, monkeypatch):
        """test that function ask_play returns a bool"""
        board = [["   " for _ in range(6)] for _ in range(6)]
        board[2][2] = "T_X"
        board[3][3] = "T_O"
        oxo.set_player1("Bobby")
        monkeypatch.setattr("sys.stdin", StringIO("XC2C1\n"))
        #monkeypatch.setattr('builtins.input', lambda _: "XC2C1")
        assert isinstance(oxo.ask_play(board, "Bobby", "Roby"), str)

    def test_is_winner(self):
        """test that function is_winner returns a bool"""
        board = [["   " for _ in range(6)] for _ in range(6)]
        board[2][2] = "T_X"
        board[3][3] = "T_O"
        assert isinstance(oxo.is_winner(board, "Bobby", (1,1)), bool)

class TestPlayerNames:
    ''' Test player's name correctness'''

    def test_empty_player_names(self):
        ''' Test if empty names are accepted'''
        assert oxo.valid_player_names("", "Bob") is False
        assert oxo.valid_player_names("Bobby", "") is False

    def test_lower_case_player_names(self):
        ''' Test if lower case names are accepted'''
        assert oxo.valid_player_names("vincent", "Bob") is False
        assert oxo.valid_player_names("Bobby", "daniel") is False

    def test_same_letter_player_names(self):
        ''' Test if names with same first letter are accepted'''
        assert oxo.valid_player_names("Bernard", "Bob") is False
        assert oxo.valid_player_names("Victoire", "Vincent") is False

    def test_protected_letter_player_names(self):
        ''' Test if names with T as first letter are accepted'''
        assert oxo.valid_player_names("Toto", "Bob") is False
        assert oxo.valid_player_names("Bobby", "Travis") is False

    def test_good_player_names(self):
        ''' Test if valid names are accepted'''
        assert oxo.valid_player_names("René", "Bob") is True
        assert oxo.valid_player_names("Daniel", "Vincent") is True

class TestTokenDrops:
    ''' Test the move's possibilities of a pawn'''

    def test_full_board(self):
        ''' Test when the board is full'''
        board = [['V_X' for _ in range(6)] for _ in range(6)]
        board[2][3] = 'T_O'
        assert len(oxo.all_token_drops(board, (2,3))) == 0

    def test_empty_board(self):
        ''' Test when the board is empty'''
        board = oxo.init_board()
        t_x = oxo.find_totem(board, 'T_X')
        assert len(oxo.all_token_drops(board, t_x)) == 4

    def test_corner(self):
        ''' Test in a corner of the board '''
        board = oxo.init_board()
        t_x = oxo.find_totem(board, 'T_X')
        board[t_x[0]][t_x[1]] = oxo.EMPTY_CELL
        board[0][0] = 'T_X'
        assert len(oxo.all_token_drops(board, (0,0))) == 2

    def test_classic_cases(self):
        ''' Test normal cases '''
        board = [
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', 'T_X', 'J_X', 'P_X', 'T_O', '   '],
            ['   ', 'P_X', '   ', '   ', 'J_O', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   ']
        ]
        t_x = oxo.find_totem(board, 'T_X')
        assert len(oxo.all_token_drops(board, t_x)) == 2

        board = [
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', 'P_X', 'J_X', 'P_X', 'T_O', '   '],
            ['   ', 'P_X', 'T_X', '   ', 'J_O', '   '],
            ['   ', '   ', 'J_O', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   ']
        ]
        t_x = oxo.find_totem(board, 'T_X')
        assert len(oxo.all_token_drops(board, t_x)) == 1

        board = [
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', 'P_X', 'J_X', 'P_X', 'T_O', '   '],
            ['   ', 'P_X', 'T_X', 'P_X', 'J_O', '   '],
            ['   ', '   ', 'J_O', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   ']
        ]
        t_x = oxo.find_totem(board, 'T_X')
        assert len(oxo.all_token_drops(board, t_x)) == 0

        board = [
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['T_X', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', 'T_O', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   ']
        ]
        t_x = oxo.find_totem(board, 'T_X')
        drops = oxo.all_token_drops(board, t_x)
        assert len(drops) == 3
        assert (1,0) in drops
        assert (3,0) in drops
        assert (2,1) in drops

        t_o = oxo.find_totem(board, 'T_O')
        drops = oxo.all_token_drops(board, t_o)
        assert len(drops) == 4
        assert (2,3) in drops
        assert (3,2) in drops
        assert (3,4) in drops
        assert (4,3) in drops


class TestConvertCoord:
    ''' Test convert coord from human instruction to matrix coord'''
    def test_convert_coord_incorrect(self):
        ''' Verify that exception are raised when incorrect coord '''
        try:
            oxo.convert_coord('YEAH')
            assert False
        except ValueError:
            assert True

        try:
            oxo.convert_coord('3D')
            assert False
        except ValueError:
            assert True

        try:
            oxo.convert_coord('A0')
            assert False
        except ValueError:
            assert True

        try:
            oxo.convert_coord('A9')
            assert False
        except ValueError:
            assert True

        try:
            oxo.convert_coord('X5')
            assert False
        except ValueError:
            assert True

    def test_convert_coord(self):
        ''' Test coord are correctly converted'''
        assert oxo.convert_coord('A1') == (0,0)
        assert oxo.convert_coord('A2') == (1,0)
        assert oxo.convert_coord('B2') == (1,1)
        assert oxo.convert_coord('C5') == (4,2)
        assert oxo.convert_coord('F6') == (5,5)

    def test_reverse_convert_coord(self):
        ''' Test coord are correctly converted'''
        assert oxo.reverse_convert_coord((0,0)) == ('A1')
        assert oxo.reverse_convert_coord((1,0)) == ('A2')
        assert oxo.reverse_convert_coord((1,1)) == ('B2')
        assert oxo.reverse_convert_coord((4,2)) == ('C5')
        assert oxo.reverse_convert_coord((5,5)) == ('F6')

class TestAction:
    ''' Tests for action reading and correctness'''

    def test_incorrect_action(self):
        ''' Incorrect actions '''
        # Incorrect totem name
        assert oxo.is_action('AC1B1') is False

        # Too short or too long
        assert oxo.is_action('C2') is False
        assert oxo.is_action('XC1') is False
        assert oxo.is_action('XC1B') is False
        assert oxo.is_action('XC1B1D2') is False

        # Incorrect column
        assert oxo.is_action('XX1B1') is False
        assert oxo.is_action('XC1K1') is False

        # Incorrect row
        assert oxo.is_action('XC9B1') is False
        assert oxo.is_action('XC1B8') is False


    def test_correct_action(self):
        ''' Correct actions'''
        assert oxo.is_action('XC1B1') is True
        assert oxo.is_action('OC1B1') is True
        assert oxo.is_action('XA2D4') is True
        assert oxo.is_action('OF6E5') is True

    def test_is_valid_action_incorrect(self):
        ''' Try to validate impossible actions'''
        board = oxo.init_board()
        # Incorrect action
        assert oxo.is_valid_action(board, "action", "Player1") is False
        # Full board
        board = [['P_X' for _ in range(6)] for _ in range(6)]
        board[2][3] = 'T_X'
        assert oxo.is_valid_action(board, "XC1B1", "Player1") is False
        # Totem move impossible
        board = [
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', 'T_X', 'J_X', 'P_X', 'T_O', '   '],
            ['   ', 'P_X', '   ', '   ', 'J_O', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   ']
        ]
        assert oxo.is_valid_action(board, "XC3C4", "Player1") is False
        assert oxo.is_valid_action(board, "XB1B2", "Player1") is True


    def test_is_valid_action_correct(self):
        ''' Try to validate possible actions'''
        # First move
        board = [
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', 'T_X', '   ', '   ', '   '],
            ['   ', '   ', '   ', 'T_O', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   ']
        ]
        #assert oxo.is_valid_action(board, "OC4D4", "Player1") is True

        board = [
            ['   ', 'J_O', 'T_X', 'P_X', '   ', '   '],
            ['   ', 'J_X', 'J_O', 'J_O', '   ', 'P_X'],
            ['   ', 'P_X', 'J_X', '   ', 'T_O', 'P_X'],
            ['   ', 'P_X', '   ', 'J_O', 'J_O', 'J_O'],
            ['   ', 'J_O', 'J_O', 'P_X', '   ', '   '],
            ['   ', 'J_O', '   ', 'P_X', '   ', '   ']
        ]
        assert oxo.is_valid_action(board, "XC4A1", "Player1") is True

        board = [
            ['P_X', '   ', 'P_O', '   ', 'P_O', '   '],
            ['X_X', 'T_X', 'P_O', 'T_O', '   ', '   '],
            ['   ', 'X_X', '   ', '   ', 'X_O', '   '],
            ['   ', '   ', '   ', 'P_O', '   ', 'P_O'],
            ['X_X', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', 'X_X', '   ', 'X_X', '   ']
        ]
        t_x = oxo.find_totem(board, 'T_X')
        assert oxo.is_valid_action(board, "XB1B2", "Player1") is True

class TestTotemMoves:
    '''Test the possible moves of the totem'''

    def test_find_totem(self):
        ''' Function to find the totem'''
        board = oxo.init_board()
        # Test an exception is raised
        try:
            oxo.find_totem(board, "WRONG_TOTEM")
            assert False
        except ValueError:
            assert True
        try:
            t_x = oxo.find_totem(board, 'T_X')
            assert board[t_x[0]][t_x[1]] == 'T_X'
            t_o = oxo.find_totem(board, 'T_O')
            assert board[t_o[0]][t_o[1]] == 'T_O'
        except TotemException:
            assert False

    def test_init_position(self):
        ''' Test that totems are correctly placed '''
        board = oxo.init_board()
        t_x = oxo.find_totem(board, 'T_X')
        t_o = oxo.find_totem(board, 'T_O')
        assert (t_x[0] == 2 or t_x[0] == 3) and (t_x[1] == 2 or t_x[1] == 3)
        assert (t_o[0] == 2 or t_o[0] == 3) and (t_o[1] == 2 or t_o[1] == 3)
        assert t_x != t_o

    def test_first_move_after_init(self):
        ''' Test that the whole row and line is allowed for a totem after the board init '''
        board = oxo.init_board()
        moves = oxo.all_totem_moves(board, 'T_X')
        assert len(moves) == 10

    def test_no_move_possible(self):
        ''' Test a totem can't move if the board is full'''
        board = [['NO!' for _ in range(6)] for _ in range(6)]
        board[2][3] = 'T_X'
        assert oxo.all_totem_moves(board, 'T_X') == set()

    def test_totem_near_pawn_jump_right_down(self):
        ''' Test totem jumps '''
        board = [
            ['D_O', 'T_O', '   ', 'P_X', '   ', '   '],
            ['D_O', 'D_O', '   ', '   ', '   ', '   '],
            ['T_X', 'P_X', '   ', '   ', '   ', '   '],
            ['D_X', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   ']
        ]
        expected_moves = {(2,2), (4,0)}
        moves = oxo.all_totem_moves(board, 'T_X')
        assert moves == expected_moves

    def test_totem_near_pawn_jump_left_up(self):
        ''' Test totem jumps '''
        board = [
            ['   ', 'T_O', '   ', 'P_X', '   ', '   '],
            ['   ', 'D_O', '   ', '   ', 'D_O ', '   '],
            ['   ', 'P_X', 'P_X', 'P_X', 'T_X', 'P_X'],
            ['D_X', '   ', '   ', '   ', 'D_O', '   '],
            ['   ', '   ', '   ', '   ', 'D_O', '   '],
            ['   ', '   ', '   ', '   ', 'D_O', '   ']
        ]
        expected_moves = {(0, 4), (2,0)}
        moves = oxo.all_totem_moves(board, 'T_X')
        assert moves == expected_moves
    
    def test_some_moves(self):
        board = [
            ['T_X', '   ', '   ', 'P_X', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   ']
        ]
        expected_moves = {(0, 1), (0, 2),  # horizontal moves
                          (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)}  # vertical moves
        moves = oxo.all_totem_moves(board, 'T_X')
        assert moves == expected_moves

    def test_move_totem(self):
        ''' Test to move the totem '''
        board = [
            ['   ', 'T_O', '   ', 'P_X', '   ', '   '],
            ['   ', 'D_O', '   ', '   ', 'D_O ','   '],
            ['   ', 'P_X', '   ', 'P_X', 'T_X', 'P_X'],
            ['D_X', '   ', '   ', '   ', 'D_O', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', 'D_O', '   ']
        ]
        oxo.move_totem(board, "T_O", (1,2))
        assert board[1][2]=="T_O"
        assert board[0][1]=="   "

        oxo.move_totem(board, "T_X", (5,0))
        assert board[5][0]=="T_X"
        assert board[2][4]=="   "


        board = [
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', 'T_X', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', 'T_O', '   ', '   ', '   ', '   ']
        ]
        oxo.move_totem(board, "T_X", (4,1))
        assert board[4][1]=="T_X"
        assert board[1][4]=="   "

        oxo.move_totem(board, "T_O", (1,5))
        assert board[1][5]=="T_O"
        assert board[5][1]=="   "

    def test_is_landlocked(self):
        ''' Tests when the totem is surrounded '''
        board = [
            ['   ', 'T_O', '   ', 'P_X', '   ', '   '],
            ['   ', 'D_O', '   ', '   ', 'D_O ','   '],
            ['   ', 'P_X', '   ', 'P_X', 'T_X', 'P_X'],
            ['D_X', '   ', '   ', '   ', 'D_O', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', 'D_O', '   ']
        ]
        totem_coords = oxo.find_totem(board, 'T_X')
        assert oxo.is_landlocked(board, totem_coords) is True

        board = [
            ['   ', 'T_O', '   ', 'P_X', 'D_O', 'D_O'],
            ['   ', 'D_O', '   ', '   ', 'D_O', 'D_O'],
            ['D_O', 'P_X', 'P_X', 'P_X', 'P_X', 'T_X'],
            ['D_X', '   ', '   ', '   ', 'D_O', 'D_O'],
            ['   ', '   ', '   ', '   ', 'D_O', 'D_O'],
            ['   ', '   ', '   ', '   ', 'D_O', 'D_O']
        ]
        totem_coords = oxo.find_totem(board, 'T_X')
        assert oxo.is_landlocked(board, totem_coords) is True

        board = [
            ['D_O', 'T_O', '   ', 'P_X', '   ', 'D_O'],
            ['   ', 'D_O', '   ', '   ', '   ', 'D_O'],
            ['T_O', 'P_X', 'P_X', 'P_X', '   ', 'P_X'],
            ['D_X', '   ', '   ', '   ', 'D_O', 'D_O'],
            ['D_O', '   ', '   ', '   ', 'D_O', 'D_O'],
            ['D_O', '   ', '   ', '   ', 'D_O', 'D_O']
        ]
        totem_coords = oxo.find_totem(board, 'T_O')
        assert oxo.is_landlocked(board, totem_coords) is False

        board = [
            ['D_O', 'T_O', '   ', 'P_X', '   ', 'D_O'],
            ['   ', 'D_O', '   ', '   ', '   ', 'D_O'],
            ['D_O', 'P_X', 'T_O', 'P_X', '   ', 'P_X'],
            ['D_X', '   ', '   ', '   ', 'D_O', 'D_O'],
            ['D_O', '   ', '   ', '   ', 'D_O', 'D_O'],
            ['D_O', '   ', '   ', '   ', 'D_O', 'D_O']
        ]
        totem_coords = oxo.find_totem(board, 'T_O')
        assert oxo.is_landlocked(board, totem_coords) is False

    def test_is_totem_fully_landlocked(self):
        ''' Tests when the totem is fully landlocked '''
        board = [
            ['   ', 'T_O', '   ', 'P_X', '   ', '   '],
            ['   ', 'D_O', '   ', '   ', 'D_O ', '   '],
            ['   ', 'P_X', 'P_X', 'P_X', 'T_X', 'P_X'],
            ['D_X', '   ', '   ', '   ', 'D_O', '   '],
            ['   ', '   ', '   ', '   ', 'D_O', '   '],
            ['   ', '   ', '   ', '   ', 'D_O', '   ']
        ]
        totem_coords = oxo.find_totem(board, 'T_X')
        assert oxo.is_totem_fully_landlocked(board, totem_coords) is False

        board = [
            ['   ', 'T_O', '   ', 'P_X', 'D_O', '   '],
            ['   ', 'D_O', '   ', '   ', 'D_O ', '   '],
            ['D_O', 'P_X', 'P_X', 'P_X', 'T_X', 'P_X'],
            ['D_X', '   ', '   ', '   ', 'D_O', '   '],
            ['   ', '   ', '   ', '   ', 'D_O', '   '],
            ['   ', '   ', '   ', '   ', 'D_O', '   ']
        ]
        totem_coords = oxo.find_totem(board, 'T_X')
        assert oxo.is_totem_fully_landlocked(board, totem_coords) is True

        board = [
            ['   ', 'T_O', '   ', 'P_X', 'D_O', 'D_O'],
            ['   ', 'D_O', '   ', '   ', 'D_O', 'D_O'],
            ['D_O', 'P_X', 'P_X', 'P_X', 'P_X', 'T_X'],
            ['D_X', '   ', '   ', '   ', 'D_O', 'D_O'],
            ['   ', '   ', '   ', '   ', 'D_O', 'D_O'],
            ['   ', '   ', '   ', '   ', 'D_O', 'D_O']
        ]
        totem_coords = oxo.find_totem(board, 'T_X')
        assert oxo.is_totem_fully_landlocked(board, totem_coords) is True

        board = [
            ['D_O', 'T_O', '   ', 'P_X', '   ', 'D_O'],
            ['D_O', 'D_O', '   ', '   ', '   ', 'D_O'],
            ['T_O', 'P_X', 'P_X', 'P_X', '   ', 'P_X'],
            ['D_X', '   ', '   ', '   ', 'D_O', 'D_O'],
            ['D_O', '   ', '   ', '   ', 'D_O', 'D_O'],
            ['D_O', '   ', '   ', '   ', 'D_O', 'D_O']
        ]
        totem_coords = oxo.find_totem(board, 'T_O')
        assert oxo.is_totem_fully_landlocked(board, totem_coords) is False

class TestBoard:
    ''' Tests of the board'''

    def test_init_board(self):
        ''' Test the board's initialization '''
        # Test the size
        board = oxo.init_board()
        assert len(board) == 6 and len(board[0]) == 6

        # Test the totems positions, a lot of times as it's random
        for i in range(0,1000):
            t_x = oxo.find_totem(board, 'T_X')
            assert t_x in [(2,3),(3,2),(2,2),(3,3)]
            t_o = oxo.find_totem(board, 'T_O')
            assert t_o in [(2,3),(3,2),(2,2),(3,3)]
            assert t_x[0] + t_o[0] == 5
            assert t_x[1] + t_o[1] == 5

#     def test_str_board(self):
#         expected_board = "    A   B   C   D   E   F \n\
#   ╭───┬───┬───┬───┬───┬───╮\n \
# 1 |   |   |   |   |   |   │\n \
#   ├───┼───┼───┼───┼───┼───┤\n\
# 2 |   |   |   |   |   |   │\n \
#   ├───┼───┼───┼───┼───┼───┤\n \
# 3 |   |   |T_X|   |   |   │\n \
#   ├───┼───┼───┼───┼───┼───┤\n \
# 4 |   |   |   |T_O|   |   │\n \
#   ├───┼───┼───┼───┼───┼───┤\n \
# 5 |   |   |   |   |   |   │\n \
#   ├───┼───┼───┼───┼───┼───┤\n \
# 6 |   |   |   |   |   |   │\n \
#   ╰───┴───┴───┴───┴───┴───╯"
#         board = oxo.init_board()
#         oxo.move_totem(board, 'T_X', (2,2))
#         oxo.move_totem(board, 'T_O', (3,3))
#         assert oxo.str_board(board) == expected_board

    def test_nb_token(self):
        ''' Count the tokens '''
        board = [
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', 'T_X', 'J_X', 'P_X', 'T_O', '   '],
            ['   ', 'P_X', '   ', '   ', 'J_O', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   ']
        ]
        expected_answers = [6,7,7,8,8]
        answers=[]
        for token in ['P_X', 'J_X', 'J_O', 'P_O', 'F_O']:
            answers.append(oxo.nb_token(board, token))
        assert answers==expected_answers

        board = [
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', 'T_X', 'P_X', 'P_X', 'P_X', '   '],
            ['   ', 'P_X', 'P_X', 'P_X', 'P_X', '   '],
            ['   ', 'P_X', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   ']
        ]
        assert oxo.nb_token(board, 'P_X') == 0

    def test_ask_play(self, monkeypatch):
        ''' Simulate the answer of a human player '''
        board = [
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', 'T_X', 'J_X', 'P_X', 'T_O', '   '],
            ['   ', 'P_X', '   ', '   ', 'J_O', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   ']
        ]
        action =  "XB1C1"
        #monkeypatch.setattr('builtins.input', lambda _: action)
        monkeypatch.setattr("sys.stdin", StringIO(f"{action}\n"))
        oxo.set_player1("Player1")
        assert oxo.ask_play(board, "Player1", "Alice") == action

    def test_all_free_cells(self):
        ''' Test all free cells are correctly detected '''
        board = [
            ['   ', '   ', 'A_O', 'A_O', '   ', 'B_X'],
            ['   ', 'A_O', '   ', 'A_O', '   ', 'B_X'],
            ['   ', 'T_X', 'A_X', 'B_X', 'T_O', 'B_O'],
            ['A_X', 'A_X', 'A_X', 'A_X', 'B_O', 'B_O'],
            ['   ', 'B_X', 'B_O', 'B_O', '   ', '   '],
            ['   ', '   ', 'A_O', 'A_X', '   ', '   ']
        ]
        expected_cells = {(0,0), (0,1), (0,4), (1,0), (1,2), (1,4), (2,0),
                          (4,0), (4,4), (4,5), (5,0), (5,1), (5,4), (5,5)}
        free_cells = oxo.all_free_cells(board)
        assert expected_cells == free_cells

class TestEndGame:
    ''' Tests for winning or draw '''

    def test_color_winning(self):
        ''' A wins with its color (= letter A)'''
        board = [
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', 'T_X', 'A_X', 'B_X', 'T_O', '   '],
            ['A_O', 'A_X', 'A_X', 'A_O', 'B_O', '   '],
            ['   ', 'B_X', 'B_O', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   ']
        ]
        assert oxo.is_winner(board, "Alphonse", (4,1)) is False
        assert oxo.is_winner(board, "Brigitte", (4,1)) is False
        assert oxo.is_winner(board, "Alphonse", (3,0)) is True
        assert oxo.is_winner(board, "Alphonse", (3,1)) is True
        assert oxo.is_winner(board, "Alphonse", (3,2)) is True
        assert oxo.is_winner(board, "Alphonse", (3,3)) is True

        board = [
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   '],
            ['   ', 'T_X', 'A_X', 'B_X', 'T_O', '   '],
            ['A_O', '   ', 'A_X', 'A_O', 'B_O', '   '],
            ['   ', 'B_X', 'A_O', '   ', '   ', '   '],
            ['   ', '   ', 'A_X', '   ', '   ', '   ']
        ]
        assert oxo.is_winner(board, "Alphonse", (4,2)) is True


    def test_symbol_winning(self):
        ''' A wins with 4 symbols'''
        board = [
            ['   ', '   ', '   ', '   ', '   ', 'B_X'],
            ['   ', '   ', '   ', '   ', '   ', 'B_X'],
            ['   ', 'T_X', 'A_X', 'B_X', 'T_O', 'B_O'],
            ['A_X', 'A_X', 'A_X', 'A_X', 'B_O', 'B_O'],
            ['   ', 'B_X', 'B_O', 'B_O', '   ', '   '],
            ['   ', '   ', '   ', '   ', '   ', '   ']
        ]
        assert oxo.is_winner(board, "Alphonse", (4,1)) is False
        assert oxo.is_winner(board, "Brigitte", (4,1)) is False
        assert oxo.is_winner(board, "Alphonse", (3,0)) is True
        assert oxo.is_winner(board, "Alphonse", (3,1)) is True
        assert oxo.is_winner(board, "Alphonse", (3,2)) is True
        assert oxo.is_winner(board, "Alphonse", (3,3)) is True
        assert oxo.is_winner(board, "Brigitte", (2,5)) is True

        board = [
            ['   ', '   ', '   ', '   ', '   ', 'B_X'],
            ['   ', '   ', '   ', '   ', '   ', 'B_X'],
            ['   ', 'T_X', 'A_X', 'B_X', 'T_O', 'B_O'],
            ['   ', '   ', '   ', '   ', 'B_O', 'B_O'],
            ['   ', 'B_X', 'B_O', 'B_O', '   ', '   '],
            ['A_X', 'A_X', 'A_X', 'A_X', '   ', '   ']
        ]
        assert oxo.is_winner(board, "Alphonse", (5,0)) is True

        board = [
            ['   ', '   ', '   ', '   ', '   ', 'B_X'],
            ['   ', '   ', '   ', '   ', '   ', 'B_X'],
            ['   ', 'T_X', 'A_X', 'B_X', 'T_O', 'B_O'],
            ['   ', '   ', '   ', '   ', 'B_O', 'B_O'],
            ['   ', 'B_X', 'B_O', 'B_O', '   ', '   '],
            ['   ', '   ', 'A_X', 'A_X', 'A_X', 'A_X']
        ]
        assert oxo.is_winner(board, "Alphonse", (5,5)) is True

    def test_not_winning(self):
        board = [
            ['   ', '   ', '   ', '   ', '   ', 'B_X'],
            ['   ', '   ', '   ', '   ', '   ', 'B_X'],
            ['   ', 'A_X', 'A_X', 'B_X', 'T_O', 'A_O'],
            ['   ', '   ', '   ', '   ', 'B_O', 'B_O'],
            ['   ', 'B_X', 'B_O', 'B_O', '   ', '   '],
            ['   ', '   ', 'A_X', 'T_X', 'A_X', 'A_X']
        ]
        assert oxo.is_winner(board, "Alphonse", (5,5)) is False