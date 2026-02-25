# CLAUDE.md — Connect 4 RL Agent: Complete Development Guide

> **This file is the single source of truth for Claude Code when developing this project.** It contains everything needed: project context, theory, architectural decisions, implementation specs, development iterations with deliverables and test cases, training parameters, deployment details, references, and coding standards. 

---

## Table of Contents

1. [Project Overview & Context](#1-project-overview--context)
2. [Algorithm Theory](#2-algorithm-theory)
3. [Architecture Decisions & Rationale](#3-architecture-decisions--rationale)
4. [Neural Network Specification](#4-neural-network-specification)
5. [MCTS Specification](#5-mcts-specification)
6. [Training Strategy & Parameters](#6-training-strategy--parameters)
7. [Repository Structure](#7-repository-structure)
8. [Coding Standards](#8-coding-standards)
9. [Development Iterations](#9-development-iterations)
10. [Testing Methodology](#10-testing-methodology)
11. [ONNX Export & Kaggle Integration](#11-onnx-export--kaggle-integration)
12. [Web Deployment](#12-web-deployment)
13. [Common Pitfalls & Solutions](#13-common-pitfalls--solutions)
14. [Performance Optimization](#14-performance-optimization)
15. [Reference Papers & Projects](#15-reference-papers--projects)
16. [Dependencies & Commands](#16-dependencies--commands)

---

## 1. Project Overview & Context

### What We're Building

A Connect 4 AI agent trained via **AlphaZero-style self-play reinforcement learning** — combining Monte Carlo Tree Search (MCTS) with a dual-headed residual neural network. The agent will:

- Train entirely from self-play (no human game data, no game knowledge beyond rules)
- Be competitive on Kaggle's Connect-X leaderboard
- Be playable in-browser via a custom web interface (zero server cost, fully client-side)
- Serve as a deep learning education project

### Developer Context

- Has already built a Connect 4 AI using classical search (minimax, alpha-beta pruning, iterative deepening, transposition tables) for Kaggle's Connect-X competition
- Comfortable with Python/NumPy but new to deep RL and neural networks
- Using Claude Code (Cursor IDE) as coding assistant
- Hardware: MacBook for development, cloud GPU (Vast.ai/RunPod/Colab) for training
- Budget: <$50 for GPU training

### Honest Performance Expectations

Connect 4 was solved in 1988 (first player wins with center opening under perfect play). A well-optimized alpha-beta solver achieves **perfect play** in real-time. No RL agent has matched this. This project's value is as a deep learning education exercise that also produces a genuinely strong player.

Realistic benchmarks after full training:
- Beats minimax depth-5 in majority of games
- Competitive for Kaggle top 10-20%
- Still makes errors on hardest positions that a perfect solver handles
- The trained value network can potentially serve as an evaluation function inside the developer's existing alpha-beta engine (hybrid approach)

---

## 2. Algorithm Theory

### 2.1 AlphaZero Overview

AlphaZero combines two ideas:

1. **Monte Carlo Tree Search (MCTS):** A planning algorithm that builds a search tree by running simulated games forward. Balances exploration and exploitation using the PUCT formula.

2. **Neural Network as Intuition:** A network that takes a board position and outputs (a) a **policy** — probability distribution over moves, and (b) a **value** — who's winning (-1 to +1). This replaces MCTS's traditional random rollouts with learned evaluation.

### 2.2 The Training Loop

```
Repeat for N iterations:
  1. SELF-PLAY: Current network guides MCTS to play games against itself
     - MCTS visit counts → policy training targets
     - Game outcomes (+1/-1/0) → value training targets
  2. TRAIN: Update network on collected (state, policy, value) data
  3. EVALUATE: New network plays 128+ games vs previous best
     - If wins >55% → becomes the new best
     - If not → keep the old best (prevents regression)
```

### 2.3 MCTS with Neural Network Guidance

Each MCTS simulation follows four steps:

**SELECT:** From root, traverse the tree using PUCT until reaching an unexpanded node:
```
a* = argmax_a [ Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a)) ]
```
- `Q(s,a)` = W(s,a) / N(s,a) = mean value of taking action a from state s
- `P(s,a)` = neural network's prior probability for action a
- `N(s,a)` = visit count for this state-action pair
- `c_puct` = 2.0 (exploration constant)

**EXPAND:** Add the new node to the tree. Run the neural network on this position to get policy priors P and value estimate v.

**EVALUATE:** The value v from the neural network replaces random rollouts. This is the key AlphaZero innovation.

**BACKUP:** Propagate v back up the path, updating Q values and visit counts. **Flip the sign at each level** (opponent's gain is our loss).

After all simulations, the move with the most visits is selected.

### 2.4 Self-Play Training Paradigm

**Why self-play works for two-player zero-sum games:**
- The game is symmetric — anything player 1 learns applies to player 2
- The opponent is always at your current skill level — natural curriculum
- No human data needed, avoiding human biases

**Preventing non-stationarity:**
- Large replay buffer mixing data from multiple iterations
- Arena evaluation with conservative 55% threshold prevents regression
- Dirichlet noise at MCTS root (ε=0.25, α=1.0) maintains exploration
- Temperature schedule: τ=1.0 for first 20 moves (explore), then τ=0.3 (exploit)

### 2.5 Key Differences from Classical Search

| Classical (Minimax) | AlphaZero (MCTS+NN) |
|---|---|
| Full-width search to fixed depth | Selective deep search guided by neural network |
| Hand-crafted evaluation function | Learned evaluation (value head) |
| Alpha-beta pruning eliminates branches | PUCT formula focuses on promising branches |
| Deterministic best move | Stochastic during training, deterministic in competition |
| Transposition table caches positions | MCTS tree caches search within one move |
| Guaranteed optimal within search depth | Probabilistically finds good moves, may miss forced wins |

---

## 3. Architecture Decisions & Rationale

### 3.1 Board Representation

**Input encoding:** 3 binary planes of shape `(3, 6, 7)`:
- Plane 0: Current player's pieces (1 where piece exists, 0 elsewhere)
- Plane 1: Opponent's pieces
- Plane 2: Constant turn indicator (all 1s if Player 1, all 0s if Player 2)

**Always encode from current player's perspective.** Before feeding to the network, rotate the board so the current player is always "player 1" from the network's viewpoint. This halves the function the network needs to learn.

**No move history needed.** Unlike chess (repetition rules) or Go (ko), Connect 4's current position contains all relevant information.

### 3.2 Game Engine: Bitboard

Use two Python `int` values (arbitrary-precision) to represent each player's pieces.

```
Board layout (bit positions):
  5 12 19 26 33 40 47
  4 11 18 25 32 39 46
  3 10 17 24 31 38 45
  2  9 16 23 30 37 44
  1  8 15 22 29 36 43
  0  7 14 21 28 35 42

Each column uses 7 bits (6 rows + 1 sentinel bit at top).
Total: 49 bits per player.

Win detection via shifting and AND-ing:
  Horizontal: shift by 7
  Vertical: shift by 1
  Diagonal \: shift by 6
  Diagonal /: shift by 8
```

Bitboard enables O(1) win detection, O(1) legal move generation, and fast cloning for MCTS simulations. This is critical — MCTS runs hundreds of game simulations per move.

### 3.3 Data Augmentation

Connect 4 boards are symmetric under horizontal reflection. Every training sample `(state, policy, value)` generates a second sample by flipping the board and policy vector left-to-right. This doubles training data for free.

### 3.4 Why ResNet Over Alternatives

- **Plain CNN:** Gradient degradation in deeper networks. Skip connections solve this.
- **Transformers:** Vision Transformers performed worse than CNNs in AlphaZero for chess (AlphaVile paper, 2023-2024). Connect 4's 7×6 grid is far too small for attention mechanisms to add value.
- **ResNet ✅:** Standard for AlphaZero. Skip connections enable deeper networks. Proven on Connect 4 across multiple implementations.

---

## 4. Neural Network Specification

### Architecture: ResNet with Dual Policy + Value Heads

```
Input: (batch, 3, 6, 7) float32 tensor

Shared trunk:
  Initial block:
    Conv2d(in=3, out=128, kernel=3×3, padding=1) → BatchNorm2d(128) → ReLU

  5× Residual Block (each block):
    Conv2d(128→128, 3×3, padding=1) → BatchNorm2d(128) → ReLU
    Conv2d(128→128, 3×3, padding=1) → BatchNorm2d(128)
    + skip connection from block input → ReLU

Policy head (move probabilities):
  Conv2d(128→32, kernel=1×1) → BatchNorm2d(32) → ReLU
  Flatten → Linear(32 × 6 × 7 = 1344, out=7)
  Output: raw logits (NOT softmaxed — softmax applied after illegal move masking)

Value head (position evaluation):
  Conv2d(128→32, kernel=1×1) → BatchNorm2d(32) → ReLU
  Flatten → Linear(1344, out=256) → ReLU → Linear(256, out=1) → Tanh
  Output: scalar in [-1, +1]
```

**Total: ~1.6 million parameters, ~6MB model file.**

### Configurable Architecture

The network must be config-driven for testing with smaller models:

| Config | Blocks | Filters | ~Params | Use |
|---|---|---|---|---|
| `tiny.yaml` | 2 | 32 | ~100K | Unit tests, code validation |
| `small.yaml` | 3 | 64 | ~400K | Quick training runs |
| `full.yaml` | 5 | 128 | ~1.6M | Production training |

### Loss Function

```
L = MSE(v_predicted, z_actual) + CrossEntropy(π_predicted, π_mcts) + c * ||θ||²
```
- `z_actual` ∈ {-1, 0, +1} — game outcome from current player's perspective
- `π_mcts` — MCTS visit count distribution, normalized to sum to 1.0
- `c = 1e-4` — L2 regularization (weight decay)

### Legal Move Masking

Legal move masking is applied **outside** the network, not as a layer:

```python
# Set illegal move logits to -inf BEFORE softmax
policy_logits[~legal_moves_mask] = float('-inf')
policy = F.softmax(policy_logits, dim=-1)
```

This ensures illegal moves get exactly zero probability.

### Training Hyperparameters

- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 1024
- Weight decay: 1e-4
- Replay buffer: FIFO, max 1,000,000 samples
- Training epochs per iteration: 10

---

## 5. MCTS Specification

### Data Structures

```python
class MCTSNode:
    """A node in the MCTS search tree.

    Attributes:
        parent: Parent node (None for root)
        action: The move (column 0-6) that led to this node
        prior: P(s,a) from neural network policy
        visit_count: N(s,a) — times this node was visited
        value_sum: W(s,a) — total value accumulated
        children: dict mapping action (int) → MCTSNode
        board: Connect4Board at this position
        is_expanded: bool — whether children have been created
    """

    @property
    def q_value(self) -> float:
        """Q(s,a) = W(s,a) / N(s,a). Return 0.0 if unvisited."""
```

### PUCT Selection

```python
def _puct_score(self, parent: MCTSNode, child: MCTSNode) -> float:
    """
    UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))
    """
    exploration = self.c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    return child.q_value + exploration
```

### Dirichlet Noise (Root Only)

```python
# Before search, add noise to root node's children priors:
noise = np.random.dirichlet([alpha] * num_legal_moves)
for i, child in enumerate(root.children.values()):
    child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]
# alpha = 1.0 (scaled for ~7 legal moves in Connect 4)
# epsilon = 0.25
```

### Temperature-Controlled Move Selection

```python
def select_move(visit_counts: np.ndarray, temperature: float) -> int:
    """
    temperature ≈ 0: argmax (deterministic, picks most-visited)
    temperature = 1: sample proportionally to visit counts
    """
    if temperature < 0.01:
        return int(np.argmax(visit_counts))
    adjusted = visit_counts ** (1.0 / temperature)
    probs = adjusted / adjusted.sum()
    return int(np.random.choice(len(probs), p=probs))
```

### Simulation Budget by Context

| Context | Simulations | Rationale |
|---|---|---|
| Self-play training | 600 | Quality training data |
| Arena evaluation | 200 | Faster, still strong |
| Kaggle competition | 100-400 | 2s time limit constraint |
| Web (policy-only) | 0 | Raw network, instant |
| Web (with MCTS) | 50-100 | Responsive feel |

### Value Backpropagation

**Critical: flip the sign at each level.** When backing up value v from a leaf:
- The leaf's parent gets -v (opponent's perspective)
- The grandparent gets +v
- And so on up to root

Getting this wrong produces an agent that plays to lose.

---

## 6. Training Strategy & Parameters

### Complete Parameter Table

| Parameter | Value | Notes |
|---|---|---|
| MCTS simulations/move (training) | 600 | Balance quality vs speed |
| MCTS simulations/move (competition) | 100-400 | Constrained by 2s limit |
| c_puct | 2.0 | Exploration constant |
| Temperature (moves 1-20) | 1.0 | Explore diverse openings |
| Temperature (moves 21+) | 0.3 | Exploit, play stronger |
| Temperature threshold move | 20 | When to switch |
| Dirichlet alpha | 1.0 | Scaled for ~7 legal moves |
| Dirichlet epsilon | 0.25 | Noise fraction at root |
| Self-play games/iteration | 5,000 | Start smaller for testing |
| Total iterations | 15-25 | Monitor convergence |
| Training epochs/iteration | 10 | Passes over replay data |
| Batch size | 1,024 | |
| Learning rate | 1e-3 | Adam optimizer |
| Weight decay | 1e-4 | L2 regularization |
| Replay buffer max size | 1,000,000 | FIFO eviction |
| Arena games | 128 | Half as P1, half as P2 |
| Update threshold | 55% | Win rate to accept new model |
| Arena temperature | 0.1 | Near-deterministic play |

### Phased Training Plan

**Phase 1 — Local Development (MacBook, free, 1-2 days):**
Build the full pipeline. Test with tiny config (2 blocks, 32 filters, 100 games/iter, 50 MCTS sims). PyTorch MPS backend works on Apple Silicon. Validate code end-to-end.

**Phase 2 — Free Cloud (Colab + Kaggle, free, 1-2 days):**
Scale to target architecture (5 blocks, 128 filters). Run 5 iterations on T4 GPU. Verify training metrics improve. Combined free tiers provide ~60 GPU hours/week.

**Phase 3 — Full Training (Vast.ai/RunPod, $10-35, 1-2 days):**
20-25 iterations, 5000 games/iteration, 600 MCTS sims/move. Estimated 30-50 GPU hours. Vast.ai RTX 3090 at $0.16-0.22/hr = $6-11 base + 50% overhead = $10-20 total.

**Phase 4 — Evaluation & Deployment (MacBook, free):**
Benchmark against classical agent. Export to ONNX. Build web interface. Submit to Kaggle.

### Training Data Format

```python
@dataclass
class TrainingSample:
    state: np.ndarray     # (3, 6, 7) float32 — board encoding
    policy: np.ndarray    # (7,) float32 — MCTS visit count distribution (sums to 1)
    value: float          # +1 (current player won), -1 (lost), 0 (draw)
```

Data augmentation: for each sample, also store the horizontally-flipped version (flipped state + flipped policy vector). This doubles dataset for free.

### Replay Buffer

FIFO with configurable max size. Oldest samples evicted when full. Mini-batches sampled uniformly at random. Mixing data from multiple iterations prevents overfitting to current play style.

### Mixed-Precision Training

Use `torch.cuda.amp` on GPU for 1.5-2× throughput:
```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    policy_logits, value = model(states)
    loss = compute_loss(policy_logits, value, target_policies, target_values)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 7. Repository Structure

```
connect4-rl/
├── CLAUDE.md                    # THIS FILE — single source of truth
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── game/                    # Connect 4 engine
│   │   ├── __init__.py
│   │   ├── board.py             # Bitboard implementation
│   │   └── constants.py         # ROWS=6, COLS=7, WIN_LENGTH=4
│   ├── neural_net/              # PyTorch model
│   │   ├── __init__.py
│   │   └── model.py             # ResNet with dual heads
│   ├── mcts/                    # Monte Carlo Tree Search
│   │   ├── __init__.py
│   │   └── search.py            # MCTS with NN evaluation
│   ├── training/                # Self-play + training loop
│   │   ├── __init__.py
│   │   ├── self_play.py         # Game generation via MCTS
│   │   ├── trainer.py           # NN training on replay buffer
│   │   ├── arena.py             # Model vs model evaluation
│   │   ├── replay_buffer.py     # FIFO replay buffer
│   │   └── coach.py             # Orchestrates full training loop
│   ├── agents/                  # Player implementations
│   │   ├── __init__.py
│   │   ├── base_agent.py        # Abstract base class
│   │   ├── random_agent.py
│   │   ├── minimax_agent.py     # For benchmarking
│   │   ├── mcts_agent.py        # Pure MCTS (no NN)
│   │   └── alphazero_agent.py   # MCTS + trained NN
│   ├── export/                  # Deployment
│   │   ├── __init__.py
│   │   ├── onnx_export.py       # PyTorch → ONNX conversion
│   │   └── kaggle_agent.py      # Self-contained Kaggle submission
│   └── utils/
│       ├── __init__.py
│       └── config.py            # Dataclass-based configuration
├── tests/                       # pytest test suite
│   ├── conftest.py              # Shared fixtures
│   ├── test_board.py
│   ├── test_model.py
│   ├── test_mcts.py
│   ├── test_training.py
│   └── test_agents.py
├── web/                         # Browser UI (built in Iteration 8)
│   ├── index.html
│   ├── style.css
│   ├── game.js                  # Game logic + UI controller
│   └── ai_worker.js             # Web Worker for ONNX inference
├── notebooks/
│   └── analysis.ipynb           # Training visualization
├── configs/
│   ├── tiny.yaml                # 2 blocks, 32 filters (testing)
│   ├── small.yaml               # 3 blocks, 64 filters (quick runs)
│   └── full.yaml                # 5 blocks, 128 filters (production)
├── checkpoints/                 # Model saves (gitignored)
├── logs/                        # Training logs (gitignored)
└── scripts/
    ├── train.py                 # python scripts/train.py --config configs/tiny.yaml
    ├── evaluate.py              # Benchmark against agents
    ├── export_onnx.py           # Export model to ONNX
    └── kaggle_submit.py         # Package for Kaggle
```

---

## 8. Coding Standards

### General Rules

- **Python 3.11+** with type hints on ALL function signatures
- **Docstrings** on every public class and method (Google style)
- **No magic numbers** — all constants in `config.py` or `constants.py`
- **Explicit over implicit** — readable code over clever one-liners
- Keep functions under 40 lines where possible; extract helpers
- Use `dataclasses` for configuration objects
- Use `pathlib.Path` instead of string path manipulation
- Use `logging` module (not `print`) for training output
- All randomness must accept a `seed` parameter for reproducibility

### Naming Conventions

```python
# Classes: PascalCase
class Connect4Board:
class ResidualBlock:
class MCTSNode:

# Functions/methods: snake_case
def make_move(self, column: int) -> "Connect4Board":
def get_legal_moves(self) -> list[int]:

# Constants: UPPER_SNAKE_CASE
ROWS = 6
COLS = 7
NUM_RESIDUAL_BLOCKS = 5

# Private methods: leading underscore
def _expand_node(self, node: MCTSNode) -> float:
```

### Documentation Standard

```python
def make_move(self, column: int) -> "Connect4Board":
    """Place the current player's piece in the specified column.

    The piece drops to the lowest available row (gravity).
    Returns a new Board instance; the original is unchanged (immutable).

    Args:
        column: Column index (0-6) to place the piece.

    Returns:
        A new Connect4Board with the piece placed and current player toggled.

    Raises:
        ValueError: If the column is full or out of range.
    """
```

### Error Handling

- Raise `ValueError` for invalid arguments (bad column, full board)
- Raise `RuntimeError` for illegal state transitions
- Use assertions only in tests, not in production code
- Validate inputs at module boundaries (public methods); skip in hot-path internals

---

## 9. Development Iterations

Build in this exact order. **Each iteration must have passing tests before proceeding.** The goal is to reach a working end-to-end system by Iteration 4, then improve from that baseline.

---

### Iteration 0: Project Scaffolding (2-3 hours)

**Goal:** Repository structure, dependencies, configuration system.

**Deliverables:**
- Directory structure as shown in Section 7
- `pyproject.toml` with project metadata and dependencies
- `requirements.txt`
- `src/utils/config.py` with dataclass-based configs:
  ```python
  @dataclass
  class GameConfig:
      rows: int = 6
      cols: int = 7
      win_length: int = 4

  @dataclass
  class ModelConfig:
      num_residual_blocks: int = 5
      num_filters: int = 128
      input_planes: int = 3

  @dataclass
  class MCTSConfig:
      num_simulations: int = 600
      c_puct: float = 2.0
      dirichlet_alpha: float = 1.0
      dirichlet_epsilon: float = 0.25
      temperature_threshold: int = 20
      temperature_high: float = 1.0
      temperature_low: float = 0.3

  @dataclass
  class TrainingConfig:
      num_iterations: int = 25
      self_play_games_per_iteration: int = 5000
      training_epochs: int = 10
      batch_size: int = 1024
      learning_rate: float = 1e-3
      weight_decay: float = 1e-4
      replay_buffer_max_size: int = 1_000_000
      arena_num_games: int = 128
      update_threshold: float = 0.55
      checkpoint_dir: str = "checkpoints"

  @dataclass
  class Config:
      game: GameConfig
      model: ModelConfig
      mcts: MCTSConfig
      training: TrainingConfig
  ```
- YAML config loading: `Config.from_yaml("configs/tiny.yaml")`
- `configs/tiny.yaml`, `configs/small.yaml`, `configs/full.yaml`
- `src/game/constants.py` with ROWS, COLS, WIN_LENGTH

**Tests:** Config loads from YAML, defaults work, invalid configs raise errors.

---

### Iteration 1: Game Engine (3-4 hours)

**Goal:** Rock-solid Connect 4 engine with bitboard representation.

**Deliverables:**
- `src/game/board.py` — `Connect4Board` class with all methods listed in Section 3.2
- Bitboard-based win detection, legal move generation, board encoding

**Unit Tests (`tests/test_board.py`):**
```
test_initial_board_is_empty
test_initial_player_is_one
test_make_move_places_piece_correctly
test_make_move_respects_gravity (piece falls to bottom)
test_make_move_stacks_pieces (second piece in same column lands on first)
test_make_move_toggles_current_player
test_make_move_returns_new_board (original unchanged — immutability)
test_cannot_move_in_full_column (raises ValueError)
test_cannot_move_in_invalid_column (raises ValueError for col < 0 or col > 6)
test_horizontal_win_detection
test_vertical_win_detection
test_diagonal_forward_win_detection (/ direction)
test_diagonal_backward_win_detection (\ direction)
test_no_false_win_on_three_in_a_row
test_draw_detection (full board, no winner)
test_legal_moves_excludes_full_columns
test_legal_moves_all_columns_on_empty_board (returns [0,1,2,3,4,5,6])
test_is_terminal_true_on_win
test_is_terminal_true_on_draw
test_is_terminal_false_during_game
test_get_result_plus_one_for_winner
test_get_result_minus_one_for_loser
test_get_result_zero_for_draw
test_encode_board_shape_is_3_6_7
test_encode_board_current_player_plane_correct
test_encode_board_opponent_plane_correct
test_encode_board_turn_indicator_plane_correct
test_encode_flipped_is_self_inverse (flip twice = original)
test_encode_flipped_mirrors_board_horizontally
test_clone_produces_independent_copy
test_clone_preserves_game_state
test_hash_equality_for_identical_boards
test_hash_differs_for_different_boards
```

---

### Iteration 2: Neural Network (2-3 hours)

**Goal:** ResNet with dual heads, verified forward/backward pass.

**Deliverables:**
- `src/neural_net/model.py` — `ResidualBlock` and `Connect4Net` classes as specified in Section 4
- `save_checkpoint(model, optimizer, path)` and `load_checkpoint(path, model_config)`

**Unit Tests (`tests/test_model.py`):**
```
test_policy_output_shape_is_batch_7
test_value_output_shape_is_batch_1
test_value_output_range_is_minus1_to_1 (tanh)
test_policy_logits_are_raw (not softmaxed — can be any real number)
test_batch_size_1_works
test_batch_size_32_works
test_tiny_config_creates_smaller_model (2 blocks, 32 filters)
test_full_config_creates_larger_model (5 blocks, 128 filters)
test_gradient_flows_to_all_parameters (no dead layers)
test_gradient_flows_through_both_heads (policy and value both get gradients)
test_masked_softmax_zeros_illegal_moves
test_masked_softmax_probabilities_sum_to_one
test_save_and_load_preserves_weights (predictions match after reload)
test_save_and_load_preserves_optimizer_state
test_batch_inference_matches_single (same input gives same output)
test_model_parameter_count_approximately_correct (within 10% of expected)
```

---

### Iteration 3: MCTS (4-5 hours)

**Goal:** Working MCTS with neural network integration.

**Deliverables:**
- `src/mcts/search.py` — `MCTSNode` and `MCTS` classes as specified in Section 5

**Unit Tests (`tests/test_mcts.py`):**
```
test_selects_winning_move_one_step (board where one move wins — must find it)
test_selects_winning_move_two_steps (forced win in 2 — should find with enough sims)
test_blocks_opponent_winning_move (opponent 1 move from win — must block)
test_visit_counts_increase_with_more_simulations
test_all_legal_moves_get_some_visits (with enough simulations)
test_illegal_moves_get_zero_visits
test_dirichlet_noise_changes_priors (root priors differ from raw network output)
test_no_noise_on_non_root_nodes
test_temperature_zero_selects_most_visited
test_temperature_one_allows_sampling (run many times, check not always same)
test_value_backpropagation_flips_sign_correctly
test_search_on_terminal_position_returns_zeros (or handles gracefully)
test_search_returns_valid_probability_distribution (sums to ~1, no negatives)
test_mcts_with_random_network_does_not_crash (integration test)
```

---

### Iteration 4: Self-Play + Training Loop — FIRST END-TO-END (4-5 hours)

**Goal:** Complete training pipeline producing v0.1 baseline. Even undertrained, this proves the full system works.

**Deliverables:**
- `src/training/replay_buffer.py` — FIFO buffer with `add()`, `sample()`, `__len__()`
- `src/training/self_play.py` — generates games via MCTS, collects training data
  - Temperature schedule: 1.0 for first 20 moves, 0.3 after
  - Applies horizontal flip augmentation
  - Assigns value labels from game outcome
- `src/training/trainer.py` — trains network on replay buffer
  - Combined loss: MSE(value) + CrossEntropy(policy) + L2 regularization
  - Returns metrics (policy_loss, value_loss, total_loss)
- `src/training/arena.py` — `pit(model_new, model_old, num_games)`
  - Half games as P1, half as P2
  - Low temperature (0.1)
  - Returns (wins, losses, draws)
- `src/training/coach.py` — orchestrates full loop
  - Iteration: self-play → train → arena → conditionally accept
  - Checkpoint saving/loading
  - Logging: iteration, losses, arena results
- `scripts/train.py` — entry point with `--config` argument

**Baseline test:** Run tiny config (2 blocks, 32 filters, 100 games/iter, 50 sims, 2-3 iterations). Must beat random agent >90%.

**Unit Tests (`tests/test_training.py`):**
```
test_replay_buffer_add_and_sample
test_replay_buffer_fifo_eviction_when_full
test_replay_buffer_sample_returns_correct_shapes
test_replay_buffer_sample_size_clamped_to_buffer_size
test_self_play_game_produces_samples
test_self_play_samples_have_correct_shapes (state: 3,6,7; policy: 7; value: float)
test_self_play_values_are_valid (only -1, 0, or +1)
test_self_play_augmentation_doubles_samples
test_trainer_one_step_reduces_loss
test_trainer_returns_loss_metrics
test_arena_returns_valid_counts (wins + losses + draws = num_games)
test_arena_symmetric_seating (each model plays both sides)
test_coach_single_iteration_completes
test_coach_checkpoint_save_and_resume
test_full_tiny_training_beats_random (integration test: 2 iterations then evaluate)
```

---

### Iteration 5: Benchmarking Framework (2-3 hours)

**Goal:** Systematic evaluation infrastructure.

**Deliverables:**
- `src/agents/base_agent.py` — abstract `Agent` with `select_move(board) -> int`
- `src/agents/random_agent.py` — uniform random from legal moves
- `src/agents/minimax_agent.py` — alpha-beta pruning, configurable depth (1, 3, 5)
- `src/agents/mcts_agent.py` — pure MCTS with random rollouts (no NN)
- `src/agents/alphazero_agent.py` — loads trained model + MCTS
- `scripts/evaluate.py` — round-robin tournament, outputs win rates

**Evaluation targets at each checkpoint:**

| Opponent | Games | Target |
|---|---|---|
| Random | 100 each side | >95% wins |
| Minimax depth 1 | 100 each side | >80% wins |
| Minimax depth 3 | 100 each side | >60% wins |
| Minimax depth 5 | 100 each side | >50% wins |
| Pure MCTS (1000 rollouts) | 50 each side | >55% wins |
| Self (first-player rate) | 200 games | 50-55% (balanced) |

**Unit Tests (`tests/test_agents.py`):**
```
test_random_agent_returns_legal_move
test_random_agent_different_moves_over_many_calls
test_minimax_depth1_blocks_immediate_win
test_minimax_depth1_takes_immediate_win
test_minimax_depth3_finds_two_step_win
test_alphazero_agent_loads_checkpoint_and_plays
test_all_agents_handle_near_terminal_board
test_evaluate_script_produces_results (integration)
```

---

### Iteration 6: Training at Scale (6-8 hours, mostly GPU waiting)

**Goal:** Full training run producing a competitive agent.

**Deliverables:**
- Full config: 5 blocks, 128 filters, 5000 games/iter, 600 MCTS sims
- 15-25 iterations on cloud GPU
- Training logs with loss curves and arena results
- Best model checkpoint
- Training visualization notebook

**Monitoring signals:**
- Policy loss and value loss should both decrease steadily
- Arena win rate >55% for accepted iterations
- Self-play first-player win rate 50-55% (balanced = good)
- If value predictions cluster near 0 → value head collapse → restart with different lr

---

### Iteration 7: ONNX Export + Kaggle (3-4 hours)

**Goal:** Deployed model on Kaggle.

**Deliverables:**
- ONNX export + dynamic quantization (see Section 11)
- Self-contained Kaggle agent file (see Section 11)
- Local testing with `kaggle-environments`
- First Kaggle submission

---

### Iteration 8: Web Interface (4-6 hours)

**Goal:** Playable Connect 4 in browser with custom UI.

**Deliverables:**
- Canvas-based board with drop animations, hover preview, win highlighting
- ONNX Runtime Web inference in Web Worker
- Human-vs-AI and AI-vs-AI modes
- Deployed to GitHub Pages
- Details in Section 12

---

### Iteration 9: Polish & Experimentation (ongoing)

Possible experiments:
- Hybrid: RL value network as eval function in minimax engine
- Hyperparameter tuning (c_puct, lr, network depth)
- Cython acceleration for MCTS
- Multiple difficulty levels in web UI
- AI-vs-AI demo mode

---

## 10. Testing Methodology

### Framework & Fixtures

Use `pytest`. Shared fixtures in `tests/conftest.py`:

```python
@pytest.fixture
def empty_board():
    return Connect4Board()

@pytest.fixture
def tiny_config():
    return Config.from_yaml("configs/tiny.yaml")

@pytest.fixture
def tiny_model(tiny_config):
    return Connect4Net(tiny_config.model)

@pytest.fixture
def board_one_move_from_win():
    """Board where current player can win by playing column 3."""
    # Set up board with 3 horizontal pieces...
    return board
```

### Coverage Targets

- Game engine: >95%
- Neural network: >90%
- MCTS: >85%
- Training pipeline: >80%

### Run Commands

```bash
pytest tests/ -v                                        # All tests
pytest tests/test_board.py -v                            # Specific module
pytest tests/ -v --cov=src --cov-report=term-missing     # With coverage
pytest tests/ -k "test_mcts" -v                          # By keyword
```

---

## 11. ONNX Export & Kaggle Integration

### ONNX Export

```python
dummy_input = torch.randn(1, 3, 6, 7)
torch.onnx.export(
    model, dummy_input, "model.onnx",
    opset_version=11,
    input_names=["board_state"],
    output_names=["policy_logits", "value"],
    dynamic_axes={"board_state": {0: "batch_size"}}
)

# Quantize for size/speed
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("model.onnx", "model_quantized.onnx", weight_type=QuantType.QUInt8)
```

Target: ~6MB unquantized → ~2MB quantized.

### Kaggle Submission Format

Single self-contained Python file with signature:
```python
def my_agent(observation, configuration):
    # observation.board → flat list of 42 ints (0=empty, 1=P1, 2=P2)
    # observation.mark → your player number (1 or 2)
    # configuration.columns=7, configuration.rows=6, configuration.inarow=4
    return column_index  # int 0-6
```

Model uploaded as Kaggle Dataset, loaded via `/kaggle/input/my-dataset/model.onnx`. Cache session in global variable.

### Time Constraints

- **2 seconds per move** + 60-second shared overage bank
- **10 seconds first call** (model loading)
- Use ONNX quantized inference, 100-200 MCTS sims, overage bank on critical openings

### Kaggle Agent Template

```python
import numpy as np
import onnxruntime as ort

_session = None

def _initialize():
    global _session
    _session = ort.InferenceSession('/kaggle/input/my-dataset/model.onnx')

def _encode_board(board_list, mark):
    board = np.array(board_list).reshape(6, 7)
    state = np.zeros((1, 3, 6, 7), dtype=np.float32)
    state[0, 0] = (board == mark).astype(np.float32)
    state[0, 1] = (board == (3 - mark)).astype(np.float32)
    state[0, 2] = 1.0 if mark == 1 else 0.0
    return state

def _get_legal_moves(board_list, cols=7):
    return [c for c in range(cols) if board_list[c] == 0]

def my_agent(observation, configuration):
    global _session
    if _session is None:
        _initialize()
    state = _encode_board(observation.board, observation.mark)
    policy_logits, value = _session.run(None, {"board_state": state})
    legal_moves = _get_legal_moves(observation.board)
    masked = np.full(7, -1e9)
    for m in legal_moves:
        masked[m] = policy_logits[0][m]
    return int(np.argmax(masked))
```

### Competitive Benchmarks

| Approach | ~Skill Rating | Leaderboard |
|---|---|---|
| Random | ~500-600 | Bottom |
| Minimax depth 5 | ~1069 | Top 54% |
| Plain MCTS | ~900-1000 | Mid-range |
| AlphaZero (NN+MCTS) | ~1100-1200+ | Top 10 |

---

## 12. Web Deployment

### Architecture: Fully Client-Side

```
GitHub Pages (free)
├── index.html
├── style.css
├── game.js             # Canvas rendering + game logic
├── ai_worker.js        # Web Worker for ONNX inference
├── model.onnx          # ~6MB trained model
└── (onnxruntime-web via CDN)
```

### Board Rendering (Canvas API)

- 7×6 grid with clean, modern visual design
- Smooth drop animation with gravity easing
- Column hover preview
- Winning four highlighted
- Responsive for mobile and desktop

### AI via Web Worker

```javascript
// ai_worker.js
import * as ort from 'onnxruntime-web';
let session = null;

self.onmessage = async (e) => {
    if (e.data.type === 'init') {
        session = await ort.InferenceSession.create('model.onnx');
        self.postMessage({ type: 'ready' });
    } else if (e.data.type === 'getMove') {
        const input = new ort.Tensor('float32', e.data.boardData, [1, 3, 6, 7]);
        const results = await session.run({ board_state: input });
        // Mask illegal moves, return argmax
        self.postMessage({ type: 'move', column: bestCol });
    }
};
```

### Game Modes

- **Human vs AI** — click column, AI responds with "thinking" indicator
- **AI vs AI** — automated play with speed controls
- **Difficulty** — vary MCTS sims: 0=instant/policy-only, 50=easy, 200=hard

### Fallback: Policy-Only Mode

If JavaScript MCTS is too complex for timeline, use raw policy head (no search). After training, the policy head alone beats minimax depth-5.

---

## 13. Common Pitfalls & Solutions

1. **Slow Python MCTS (HIGHEST RISK):** Python MCTS is often 10-100× too slow. Use bitboard engine, batch NN calls, consider Cython for hot loop. Profile before optimizing.

2. **Value Head Collapse:** Network predicts ~0 for all positions. Monitor value loss and prediction distribution. Ensure balanced P1/P2 data. Try lower learning rate.

3. **Catastrophic Forgetting:** Agent loses early-game skill. Fix: large replay buffer, shuffled mini-batches, data from multiple iterations.

4. **Reward Shaping Trap:** Do NOT add intermediate rewards. Use ONLY terminal: +1 win, -1 loss, 0 draw.

5. **Legal Move Masking Bugs:** Always mask BEFORE softmax with `-inf`. Test that illegal moves get zero probability.

6. **Perspective Confusion:** Always encode from current player's view. Negate value at each MCTS backup level. Test with known positions.

7. **Checkpoint Corruption:** Save model weights AND optimizer state. Verify checkpoint loads in fresh process before long runs.

---

## 14. Performance Optimization

### Checklist

- [ ] Bitboard game engine (O(1) win detection, O(1) legal moves)
- [ ] Batch NN inference during MCTS (collect leaves → one forward pass)
- [ ] Mixed-precision training (`torch.cuda.amp`) on GPU
- [ ] Parallel self-play workers (`multiprocessing`)
- [ ] ONNX export with dynamic quantization
- [ ] NumPy vectorization in data processing
- [ ] Profile before optimizing (`cProfile` or `py-spy`)

### Batched MCTS Inference

Instead of one NN call per simulation, collect pending leaves and batch:
```python
pending_leaves = []
for sim in range(num_simulations):
    leaf = self._select_and_expand(root)
    pending_leaves.append(leaf)
    if len(pending_leaves) >= batch_size or sim == num_simulations - 1:
        states = np.stack([l.board.encode() for l in pending_leaves])
        policies, values = model.predict_batch(states)
        for leaf, policy, value in zip(pending_leaves, policies, values):
            self._set_priors_and_backup(leaf, policy, value)
        pending_leaves = []
```

---

## 15. Reference Papers & Projects

### Essential Papers

1. **Silver et al. (2017) "Mastering the Game of Go without Human Knowledge"** (AlphaGo Zero) — The direct template. Self-play from tabula rasa. Nature paper.

2. **Silver et al. (2018) "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"** (AlphaZero) — Generalization to multiple games. https://arxiv.org/abs/1712.01815

3. **Gupta (2022) "Reinforcement Learning for ConnectX"** — AlphaZero vs MCTS vs minimax on Kaggle. Reached 9th/225. https://arxiv.org/pdf/2210.08263

4. **Anthony et al. (2017) "Thinking Fast and Slow with Deep Learning and Tree Search"** — Expert Iteration. NeurIPS 2017.

5. **Schrittwieser et al. (2020) "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"** (MuZero) — Reference only. https://arxiv.org/abs/1911.08265

6. **Taylor et al. (2024) "An Evolutionary Framework for Connect-4 as Test-Bed"** — Approach comparison. https://arxiv.org/abs/2405.16595

7. **Tian et al. (2019) "ELF OpenGo"** — Practical AlphaZero engineering. https://arxiv.org/abs/1902.04522

### Implementation References

| Repository | Description | Use |
|---|---|---|
| [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general) | Most popular general AlphaZero (PyTorch), ~4300 stars, includes Connect 4 | Primary code reference |
| [bhansconnect/fast-alphazero-general](https://github.com/bhansconnect/fast-alphazero-general) | Cython-optimized fork | Speed optimization |
| [AlphaZero.jl Connect 4 tutorial](https://jonathan-laurent.github.io/AlphaZero.jl/stable/tutorial/connect_four/) | Most thorough benchmarks vs perfect solver | Parameters & benchmarks |
| [emasab/simple-alpha-zero-connect4](https://github.com/emasab/simple-alpha-zero-connect4) | Training → TensorFlow.js browser deployment | Web deployment |
| [Zeta36/connect4-alpha-zero](https://github.com/Zeta36/connect4-alpha-zero) | AlphaGo Zero for Connect 4 (Keras) | Alternative reference |
| [blanyal/alpha-zero](https://github.com/blanyal/alpha-zero) | AlphaZero for Othello, Connect 4, TicTacToe | Multi-game |
| [cemkaraoguz/AlphaZero-Connect4](https://github.com/cemkaraoguz/AlphaZero-Connect4) | Clean Connect 4-specific implementation | Focused reference |

### Tutorials

- **David Foster** — "How to Build Your Own AlphaZero AI Using Python and Keras" ([Medium](https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188))
- **Josh Varty** — "Alpha Zero and Monte Carlo Tree Search" ([joshvarty.github.io/AlphaZero](https://joshvarty.github.io/AlphaZero/))
- **AI Singapore** — "From-scratch implementation of AlphaZero for Connect4" ([aisingapore.org](https://aisingapore.org/from-scratch-implementation-of-alphazero-for-connect4/))

### Connect 4 Game Theory

- **John Tromp's Connect Four Playground** — Perfect solver. [tromp.github.io/c4/c4.html](https://tromp.github.io/c4/c4.html)
- Solved 1988 by Allen & Allis. First player wins with center opening. ~4.5 trillion positions.

---

## 16. Dependencies & Commands

### Python (`requirements.txt`)

```
torch>=2.0
numpy>=1.24
onnx>=1.14
onnxruntime>=1.15
pyyaml>=6.0
tqdm>=4.65
matplotlib>=3.7
```

### Dev

```
pytest>=7.3
pytest-cov>=4.1
```

### Optional (Kaggle)

```
kaggle-environments>=1.12
```

### Web

```
onnxruntime-web  # CDN: https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/
```

### Commands

```bash
pip install -e ".[dev]"                                  # Install
pytest tests/ -v                                         # All tests
pytest tests/test_board.py -v                            # Module tests
pytest tests/ -v --cov=src --cov-report=term-missing     # Coverage
python scripts/train.py --config configs/tiny.yaml       # Train (local)
python scripts/train.py --config configs/full.yaml       # Train (GPU)
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --num-games 100
python scripts/export_onnx.py --checkpoint checkpoints/best_model.pt --output model.onnx
python scripts/kaggle_submit.py --model model.onnx --output submission/
```

### Git Practices

- Commit after each iteration's tests pass
- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`
- `.gitignore`: checkpoints/, logs/, __pycache__/, *.onnx (except web/model.onnx)
- Tags: `v0.1-baseline`, `v0.2-trained`, `v1.0-deployed`