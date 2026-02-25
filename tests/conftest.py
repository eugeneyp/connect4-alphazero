"""Shared pytest fixtures for the Connect 4 AlphaZero test suite."""

import pytest

from src.game.board import Connect4Board


@pytest.fixture
def empty_board() -> Connect4Board:
    """Return a fresh empty Connect4Board."""
    return Connect4Board()


@pytest.fixture
def board_with_moves():
    """Return a factory that builds a board from a sequence of column moves."""

    def _make(moves: list[int]) -> Connect4Board:
        board = Connect4Board()
        for col in moves:
            board = board.make_move(col)
        return board

    return _make
