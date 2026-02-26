"""AlphaZero training loop orchestrator.

Coordinates self-play game generation, neural network training, and
arena evaluation to iteratively improve the model.

The training loop per iteration:
  1. Self-play: current model generates games and fills the replay buffer.
  2. Train: update the network on samples from the replay buffer.
  3. Arena: new candidate vs. current best; accept if win rate >= threshold.
  4. Save checkpoint for the iteration.
"""

import copy
import logging
from pathlib import Path

import torch

from src.neural_net.model import Connect4Net
from src.training.arena import pit
from src.training.replay_buffer import ReplayBuffer
from src.training.self_play import SelfPlay
from src.training.trainer import Trainer
from src.utils.config import Config

logger = logging.getLogger(__name__)


class Coach:
    """Orchestrates the full AlphaZero self-play training loop.

    Args:
        config: Full configuration (game, model, mcts, training sub-configs).
    """

    def __init__(self, config: Config) -> None:
        """Initialize the coach.

        Args:
            config: Training configuration including model architecture,
                MCTS settings, and training hyperparameters.
        """
        self._config = config
        self._replay_buffer = ReplayBuffer(config.training.replay_buffer_max_size)

        # Create the model
        self._model = Connect4Net(
            num_blocks=config.model.num_residual_blocks,
            num_filters=config.model.num_filters,
            input_planes=config.model.input_planes,
        )

        checkpoint_dir = Path(config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir = checkpoint_dir
        self._best_checkpoint_path = checkpoint_dir / "best_model.pt"

    def train(self, start_iteration: int = 0) -> None:
        """Run the full training loop.

        Args:
            start_iteration: Iteration index to start from (0 for fresh
                training, or loaded from a checkpoint for resumption).
        """
        cfg = self._config.training
        model = self._model

        for iteration in range(start_iteration, cfg.num_iterations):
            logger.info("=== Iteration %d / %d ===", iteration + 1, cfg.num_iterations)

            # ----------------------------------------------------------------
            # 1. Self-play: generate games and fill the replay buffer
            # ----------------------------------------------------------------
            model.eval()
            self_play = SelfPlay(model, self._config.mcts)
            logger.info(
                "Self-play: generating %d games...", cfg.self_play_games_per_iteration
            )
            samples = self_play.generate_games(cfg.self_play_games_per_iteration)
            for sample in samples:
                self._replay_buffer.add(sample)
            logger.info("Replay buffer size: %d", len(self._replay_buffer))

            # ----------------------------------------------------------------
            # 2. Train: update the network on replay buffer data
            # ----------------------------------------------------------------
            trainer = Trainer(model, cfg)
            logger.info("Training for %d epochs...", cfg.training_epochs)
            metrics = trainer.train(self._replay_buffer, cfg.training_epochs)
            logger.info(
                "Losses — policy: %.4f  value: %.4f  total: %.4f",
                metrics["policy_loss"],
                metrics["value_loss"],
                metrics["total_loss"],
            )
            model.eval()

            # ----------------------------------------------------------------
            # 3. Arena: compare candidate vs. current best
            # ----------------------------------------------------------------
            candidate = copy.deepcopy(model)

            if self._best_checkpoint_path.exists():
                best = self._load_best_model(self._best_checkpoint_path)
                logger.info(
                    "Arena: %d games (candidate vs. best)...", cfg.arena_num_games
                )
                wins, losses, draws = pit(
                    candidate, best, cfg.arena_num_games, self._config.mcts
                )
                total = wins + losses + draws
                win_rate = wins / total if total > 0 else 0.0
                logger.info(
                    "Arena result — wins: %d  losses: %d  draws: %d  win_rate: %.3f",
                    wins,
                    losses,
                    draws,
                    win_rate,
                )

                if win_rate >= cfg.update_threshold:
                    logger.info("Accepted new model (win_rate %.3f >= %.3f)", win_rate, cfg.update_threshold)
                    model = candidate
                    self._model = model
                    self._save_checkpoint(
                        model, trainer.optimizer, self._best_checkpoint_path, iteration
                    )
                else:
                    logger.info(
                        "Rejected new model (win_rate %.3f < %.3f), reverting to best",
                        win_rate,
                        cfg.update_threshold,
                    )
                    model = best
                    self._model = model
            else:
                # First iteration: accept unconditionally
                logger.info("No previous best model — accepting candidate unconditionally.")
                self._model = candidate
                model = candidate
                self._save_checkpoint(
                    model, trainer.optimizer, self._best_checkpoint_path, iteration
                )

            # ----------------------------------------------------------------
            # 4. Save iteration checkpoint
            # ----------------------------------------------------------------
            iter_checkpoint_path = (
                self._checkpoint_dir / f"checkpoint_iter_{iteration:03d}.pt"
            )
            self._save_checkpoint(model, trainer.optimizer, iter_checkpoint_path, iteration)
            logger.info("Saved checkpoint: %s", iter_checkpoint_path)

    def _save_checkpoint(
        self,
        model: Connect4Net,
        optimizer: torch.optim.Adam,
        path: Path,
        iteration: int,
    ) -> None:
        """Save model + optimizer state to disk.

        Args:
            model: The neural network to save.
            optimizer: The Adam optimizer whose state to save.
            path: Destination file path.
            iteration: Current iteration index (stored in checkpoint).
        """
        torch.save(
            {
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_blocks": self._config.model.num_residual_blocks,
                "num_filters": self._config.model.num_filters,
            },
            path,
        )

    def _load_best_model(self, path: Path) -> Connect4Net:
        """Load the best model from a checkpoint file.

        Args:
            path: Path to the checkpoint file.

        Returns:
            Connect4Net loaded from the checkpoint, in eval mode.
        """
        checkpoint = torch.load(path, weights_only=False)
        num_blocks = checkpoint.get("num_blocks", self._config.model.num_residual_blocks)
        num_filters = checkpoint.get("num_filters", self._config.model.num_filters)

        model = Connect4Net(
            num_blocks=num_blocks,
            num_filters=num_filters,
            input_planes=self._config.model.input_planes,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
