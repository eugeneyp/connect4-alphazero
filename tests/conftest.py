"""Shared pytest fixtures for the Connect 4 AlphaZero test suite."""

import pytest
import torch

from src.game.board import Connect4Board
from src.neural_net.model import Connect4Net
from src.utils.config import ModelConfig


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


@pytest.fixture
def default_model_config() -> ModelConfig:
    """Return the default full-size model configuration (5 blocks, 128 filters)."""
    return ModelConfig(num_residual_blocks=5, num_filters=128, input_planes=3)


@pytest.fixture
def tiny_model_config() -> ModelConfig:
    """Return the tiny model configuration for fast tests (2 blocks, 32 filters)."""
    return ModelConfig(num_residual_blocks=2, num_filters=32, input_planes=3)


@pytest.fixture
def default_model(default_model_config: ModelConfig) -> Connect4Net:
    """Return a full-size Connect4Net in eval mode."""
    model = Connect4Net(
        num_blocks=default_model_config.num_residual_blocks,
        num_filters=default_model_config.num_filters,
        input_planes=default_model_config.input_planes,
    )
    model.eval()
    return model


@pytest.fixture
def tiny_model(tiny_model_config: ModelConfig) -> Connect4Net:
    """Return a tiny Connect4Net in eval mode."""
    model = Connect4Net(
        num_blocks=tiny_model_config.num_residual_blocks,
        num_filters=tiny_model_config.num_filters,
        input_planes=tiny_model_config.input_planes,
    )
    model.eval()
    return model


@pytest.fixture
def board_tensor() -> torch.Tensor:
    """Return a random (1, 3, 6, 7) board tensor with fixed seed."""
    torch.manual_seed(42)
    return torch.randn(1, 3, 6, 7)


@pytest.fixture
def empty_board_tensor(empty_board: Connect4Board) -> torch.Tensor:
    """Return the encoding of an empty board as a (1, 3, 6, 7) tensor."""
    # Convert via tolist() because this PyTorch build has no NumPy C bridge.
    return torch.tensor(empty_board.encode().tolist(), dtype=torch.float32).unsqueeze(0)
