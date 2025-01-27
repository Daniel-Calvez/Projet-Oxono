"""
Oxono Unit Tests
"""

from io import StringIO

import pytest

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
        assert isinstance(oxo.is_valid_action(board, "TC2C1", "Bobby"), bool)
        assert isinstance(oxo.is_valid_action(board, "XC2C1", "Bobby"), bool)

    def test_ask_play(self, monkeypatch):
        """test that function ask_play returns a bool"""
        board = [["   " for _ in range(6)] for _ in range(6)]
        board[2][2] = "T_X"
        board[3][3] = "T_O"
        monkeypatch.setattr("sys.stdin", StringIO("XC2C1\n"))
        assert isinstance(oxo.ask_play(board, "Bobby"), str)

    def test_is_winner(self):
        """test that function is_winner returns a bool"""
        board = [["   " for _ in range(6)] for _ in range(6)]
        board[2][2] = "T_X"
        board[3][3] = "T_O"
        assert isinstance(oxo.is_winner(board, "Bobby", (1,1)), bool)
