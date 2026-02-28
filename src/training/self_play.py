"""Self-play game generation for AlphaZero training.

Uses MCTS guided by the neural network to play full games and collect
(state, policy, value) training samples. Horizontal flip augmentation
is applied to double the dataset for free.

Parallel self-play: when num_workers > 1, games are distributed across
worker processes that each run CPU-only inference. The GPU remains
exclusively available to the main process for the training step.
"""

import concurrent.futures
import logging
import time

import numpy as np
import torch.multiprocessing as mp
from tqdm import tqdm

from src.game.board import Connect4Board
from src.game.constants import COLS
from src.mcts.search import MCTS, select_move
from src.neural_net.model import Connect4Net
from src.training.replay_buffer import TrainingSample
from src.utils.config import MCTSConfig, ModelConfig


# ---------------------------------------------------------------------------
# Module-level worker function (must be top-level for multiprocessing pickle)
# ---------------------------------------------------------------------------

def _self_play_worker(args: tuple) -> list[TrainingSample]:
    """Run self-play games in a subprocess using CPU-only inference.

    This function is called by each parallel worker process. It reconstructs
    the model from the provided architecture params and CPU state dict, then
    plays num_games complete games.

    Args:
        args: Tuple of (num_blocks, num_filters, input_planes,
              state_dict, mcts_config, num_games).

    Returns:
        Flat list of TrainingSample (original + horizontally flipped) from
        all games played by this worker.
    """
    num_blocks, num_filters, input_planes, state_dict, mcts_config, num_games = args

    model = Connect4Net(
        num_blocks=num_blocks,
        num_filters=num_filters,
        input_planes=input_planes,
    )
    model.load_state_dict(state_dict)
    model.eval()

    sp = SelfPlay(model, mcts_config)
    all_samples: list[TrainingSample] = []
    for _ in range(num_games):
        all_samples.extend(sp.play_game())
    return all_samples


# ---------------------------------------------------------------------------
# SelfPlay class
# ---------------------------------------------------------------------------

class SelfPlay:
    """Generates Connect 4 games via MCTS self-play.

    A single MCTS instance is created at construction and reused across
    games (the tree is not persisted between games).

    Args:
        model: Neural network used to guide MCTS. Should be in eval mode.
        mcts_config: MCTS hyperparameters (simulations, temperature, etc.).
        num_workers: Number of parallel worker processes for self-play.
            1 (default) uses the existing serial loop. >1 distributes games
            across subprocesses using CPU inference.
        model_config: Architecture parameters needed to reconstruct the model
            in worker processes. Required when num_workers > 1.
    """

    def __init__(
        self,
        model: Connect4Net,
        mcts_config: MCTSConfig,
        num_workers: int = 1,
        model_config: ModelConfig | None = None,
    ) -> None:
        if num_workers > 1 and model_config is None:
            raise ValueError("model_config is required when num_workers > 1")

        self._model = model
        self._mcts = MCTS(model, mcts_config)
        self._config = mcts_config
        self._num_workers = num_workers
        self._model_config = model_config

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

        Dispatches to parallel workers when num_workers > 1, otherwise
        runs the existing serial loop.

        Args:
            n: Number of complete games to play.

        Returns:
            Flat list of TrainingSample objects from all games.
        """
        if self._num_workers > 1:
            return self._generate_games_parallel(n)
        return self._generate_games_serial(n)

    def _generate_games_serial(self, n: int) -> list[TrainingSample]:
        """Run n games sequentially with tqdm progress bar."""
        logger = logging.getLogger(__name__)
        all_samples: list[TrainingSample] = []
        start = time.time()

        with tqdm(total=n, unit="game", desc="Self-play") as pbar:
            for i in range(n):
                all_samples.extend(self.play_game())
                pbar.update(1)
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start
                    rate = (i + 1) / elapsed
                    remaining = (n - i - 1) / rate if rate > 0 else 0
                    logger.info(
                        "Self-play: %d/%d games  %.2f games/min  ETA %.0f min",
                        i + 1, n, rate * 60, remaining / 60,
                    )

        elapsed = time.time() - start
        logger.info(
            "Self-play done: %d games in %.1f min (%.2f games/min)",
            n, elapsed / 60, n / elapsed * 60,
        )
        return all_samples

    def _generate_games_parallel(self, n: int) -> list[TrainingSample]:
        """Distribute n games across worker subprocesses using CPU inference.

        Games are split as evenly as possible across workers. Each worker
        receives a CPU copy of the model weights, plays its share of games
        independently, and returns the collected samples.

        Args:
            n: Total number of games to play across all workers.

        Returns:
            Flat list of TrainingSample objects from all games.
        """
        logger = logging.getLogger(__name__)
        cfg = self._model_config

        # CPU state dict — safe to pickle and send to worker processes
        state_dict_cpu = {k: v.cpu() for k, v in self._model.state_dict().items()}

        # Distribute games evenly; give remainder games to first workers
        base_chunk = n // self._num_workers
        remainder = n % self._num_workers
        chunks = [
            base_chunk + (1 if i < remainder else 0)
            for i in range(self._num_workers)
        ]

        args_list = [
            (
                cfg.num_residual_blocks,
                cfg.num_filters,
                cfg.input_planes,
                state_dict_cpu,
                self._config,
                chunk,
            )
            for chunk in chunks
        ]

        logger.info(
            "Parallel self-play: %d workers, %s games each",
            self._num_workers, chunks,
        )
        start = time.time()
        all_samples: list[TrainingSample] = []
        completed_games = 0

        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self._num_workers, mp_context=ctx
        ) as executor:
            futures = {
                executor.submit(_self_play_worker, args): i
                for i, args in enumerate(args_list)
            }
            for future in concurrent.futures.as_completed(futures):
                worker_idx = futures[future]
                samples = future.result()
                all_samples.extend(samples)
                completed_games += chunks[worker_idx]
                elapsed = time.time() - start
                rate = completed_games / elapsed * 60 if elapsed > 0 else 0
                logger.info(
                    "Self-play: worker %d done — %d/%d games  %.1f games/min",
                    worker_idx, completed_games, n, rate,
                )

        elapsed = time.time() - start
        logger.info(
            "Self-play done: %d games in %.1f min (%.2f games/min)",
            n, elapsed / 60, n / elapsed * 60,
        )
        return all_samples
