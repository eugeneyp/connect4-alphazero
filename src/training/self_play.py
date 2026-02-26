"""Self-play game generation for AlphaZero training.

Uses MCTS guided by the neural network to play full games and collect
(state, policy, value) training samples. Horizontal flip augmentation
is applied to double the dataset for free.
"""

import numpy as np

from src.game.board import Connect4Board
from src.game.constants import COLS
from src.mcts.search import MCTS, select_move
from src.neural_net.model import Connect4Net
from src.training.replay_buffer import TrainingSample
from src.utils.config import MCTSConfig


class SelfPlay:
    """Generates Connect 4 games via MCTS self-play.

    A single MCTS instance is created at construction and reused across
    games (the tree is not persisted between games).

    Args:
        model: Neural network used to guide MCTS. Should be in eval mode.
        mcts_config: MCTS hyperparameters (simulations, temperature, etc.).
    """

    def __init__(self, model: Connect4Net, mcts_config: MCTSConfig) -> None:
        """Initialize SelfPlay.

        Args:
            model: Trained (or random) Connect4Net for MCTS evaluation.
            mcts_config: Hyperparameters controlling MCTS search and
                temperature schedule.
        """
        self._mcts = MCTS(model, mcts_config)
        self._config = mcts_config

    def play_game(self) -> list[TrainingSample]:
        """Play a single self-play game and return training samples.

        Each position in the game produces two samples (original + flipped)
        for data augmentation.

        Returns:
            List of TrainingSample objects. If the game lasts N moves,
            returns 2*N samples.
        """
        board = Connect4Board()
        # history: (state_encoding, visit_count_policy, player_who_moved)
        history: list[tuple[np.ndarray, np.ndarray, int]] = []
        move_number = 0

        # Phase 1: play the game and record states/policies
        while not board.is_terminal():
            temp = (
                self._config.temperature_high
                if move_number < self._config.temperature_threshold
                else self._config.temperature_low
            )
            visit_counts = self._mcts.search(board, add_dirichlet_noise=True)
            history.append((board.encode(), visit_counts, board.current_player))
            col = select_move(visit_counts, temp)
            board = board.make_move(col)
            move_number += 1

        # Phase 2: assign values and create augmented samples
        winner = board.get_winner()  # None = draw
        samples: list[TrainingSample] = []

        for state, policy, player in history:
            if winner is None:
                value = 0.0
            elif player == winner:
                value = 1.0
            else:
                value = -1.0

            samples.append(TrainingSample(state=state, policy=policy, value=value))

            # Horizontal flip augmentation
            flipped_state = np.flip(state, axis=2).copy()
            flip_indices = np.arange(COLS - 1, -1, -1)  # [6, 5, 4, 3, 2, 1, 0]
            flipped_policy = policy[flip_indices]
            samples.append(
                TrainingSample(state=flipped_state, policy=flipped_policy, value=value)
            )

        return samples

    def generate_games(self, n: int) -> list[TrainingSample]:
        """Play n self-play games and aggregate all training samples.

        Args:
            n: Number of complete games to play.

        Returns:
            Flat list of TrainingSample objects from all games.
        """
        all_samples: list[TrainingSample] = []
        for _ in range(n):
            all_samples.extend(self.play_game())
        return all_samples
