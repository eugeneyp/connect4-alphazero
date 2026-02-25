# CLAUDE.md — Connect 4 RL Agent Development Guide

> **This file guides Claude Code in developing a Connect 4 AI agent using AlphaZero-style reinforcement learning.** It contains architectural decisions, coding standards, implementation details, and development workflow. Follow this document precisely when writing code for this project.

---

## Project Summary

Build a Connect 4 AI agent that:
1. Trains via AlphaZero-style self-play (MCTS + neural network)
2. Submits to Kaggle Connect-X competition
3. Runs in-browser via ONNX Runtime Web (zero server cost)

**Tech stack:** Python 3.11+, PyTorch, NumPy, ONNX, pytest. Web: vanilla JS + Canvas + ONNX Runtime Web.

---

## Repository Structure

```
connect4-rl/
├── CLAUDE.md                    # THIS FILE
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
│   │   └── coach.py             # Orchestrates full training loop
│   ├── agents/                  # Player implementations
│   │   ├── __init__.py
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
├── web/                         # Browser UI
│   ├── index.html
│   ├── style.css
│   ├── game.js                  # Game logic + UI
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
    ├── train.py                 # Entry point: python scripts/train.py --config configs/tiny.yaml
    ├── evaluate.py              # Benchmark against agents
    ├── export_onnx.py           # Export model to ONNX
    └── kaggle_submit.py         # Package for Kaggle
```

---

## Coding Standards

### General Rules

- **Python 3.11+** with type hints on all function signatures
- **Docstrings** on every public class and method (Google style)
- **No magic numbers** — all constants in `config.py` or `constants.py`
- **Explicit over implicit** — prefer readable code over clever one-liners
- Keep functions under 40 lines where possible; extract helpers
- Use `dataclasses` or `pydantic` for configuration objects
- Use `pathlib.Path` instead of string path manipulation
- Use `logging` module (not print statements) for training output
- All randomness must be seedable via a `seed` parameter for reproducibility

### Naming Conventions

```python
# Classes: PascalCase
class Connect4Board:
class ResidualBlock:
class MCTSNode:

# Functions/methods: snake_case
def make_move(self, column: int) -> "Connect4Board":
def get_legal_moves(self) -> list[int]:
def run_simulations(self, num_simulations: int) -> np.ndarray:

# Constants: UPPER_SNAKE_CASE
ROWS = 6
COLS = 7
WIN_LENGTH = 4
NUM_RESIDUAL_BLOCKS = 5
NUM_FILTERS = 128

# Private methods: leading underscore
def _expand_node(self, node: MCTSNode) -> float:
def _backup(self, path: list[MCTSNode], value: float) -> None:
```

### Documentation Standard

Every public class and method gets a Google-style docstring:

```python
def make_move(self, column: int) -> "Connect4Board":
    """Place the current player's piece in the specified column.

    The piece drops to the lowest available row in the column (gravity).
    Returns a new Board instance; the original is unchanged (immutable).

    Args:
        column: Column index (0-6) to place the piece.

    Returns:
        A new Connect4Board with the piece placed and the current player toggled.

    Raises:
        ValueError: If the column is full or out of range.
    """
```

### Error Handling

- Raise `ValueError` for invalid arguments (bad column, full board)
- Raise `RuntimeError` for illegal state transitions
- Use assertions only in tests, not in production code
- Validate inputs at module boundaries (public methods); skip in hot-path internals

### Testing Standards

- **Every module gets tests BEFORE the next module is built**
- Use `pytest` with descriptive test names: `test_horizontal_win_detection_rightmost_column`
- Use fixtures in `conftest.py` for shared objects (boards, models, configs)
- Parametrize tests where patterns repeat (e.g., win detection in 4 directions)
- Aim for >90% coverage on game engine and MCTS

---

## Implementation Details

### Game Engine (`src/game/board.py`)

**Bitboard representation.** Use two `int` values (Python's arbitrary-precision ints work fine) to represent each player's pieces on the 7×6 board.

```python
# Board layout (bit positions):
#  5 12 19 26 33 40 47
#  4 11 18 25 32 39 46
#  3 10 17 24 31 38 45
#  2  9 16 23 30 37 44
#  1  8 15 22 29 36 43
#  0  7 14 21 28 35 42
#
# Each column uses 7 bits (6 rows + 1 sentinel bit at top).
# Total: 49 bits per player.

# Win detection: check 4-in-a-row by shifting and AND-ing
# Horizontal: shift by 7, Vertical: shift by 1
# Diagonal \: shift by 6, Diagonal /: shift by 8
```

**Required methods:**

```python
class Connect4Board:
    def __init__(self):
        """Create empty board. Player 1 moves first."""

    def make_move(self, column: int) -> "Connect4Board":
        """Return new board with piece placed. Immutable."""

    def get_legal_moves(self) -> list[int]:
        """Return list of columns that are not full."""

    def is_terminal(self) -> bool:
        """Check if game is over (win or draw)."""

    def get_winner(self) -> int | None:
        """Return 1, 2, or None. None if draw or ongoing."""

    def get_result(self, player: int) -> float:
        """Return +1 (win), -1 (loss), 0 (draw) for given player."""

    def encode(self) -> np.ndarray:
        """Return (3, 6, 7) float32 array for neural network input.
        Plane 0: current player's pieces
        Plane 1: opponent's pieces
        Plane 2: constant turn indicator (all 1s if P1, all 0s if P2)
        Always from CURRENT player's perspective."""

    def encode_flipped(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (flipped_board_encoding, flipped_policy_indices) for data augmentation.
        Flips board horizontally (left ↔ right).
        Also returns index mapping for flipping the policy vector."""

    def clone(self) -> "Connect4Board":
        """Return independent deep copy."""

    @property
    def current_player(self) -> int:
        """1 or 2."""

    def __str__(self) -> str:
        """Pretty-print board. Use . for empty, X for P1, O for P2."""
```

### Neural Network (`src/neural_net/model.py`)

**Architecture: ResNet with dual policy + value heads.**

```python
class ResidualBlock(nn.Module):
    """Single residual block: Conv-BN-ReLU-Conv-BN + skip → ReLU"""

class Connect4Net(nn.Module):
    """
    Args:
        num_blocks: Number of residual blocks (default: 5)
        num_filters: Channels in residual tower (default: 128)
        input_planes: Number of input feature planes (default: 3)

    Input: (batch, 3, 6, 7) float32 tensor
    Output: (policy_logits, value)
        policy_logits: (batch, 7) — raw logits, NOT softmaxed
        value: (batch, 1) — tanh-activated, range [-1, 1]
    """
```

**Legal move masking** is applied OUTSIDE the network:

```python
def get_policy(model, board_tensor, legal_moves_mask):
    """Get move probabilities with illegal moves masked to zero.

    Args:
        model: The Connect4Net
        board_tensor: (1, 3, 6, 7) encoded board state
        legal_moves_mask: (7,) boolean array, True for legal columns

    Returns:
        (7,) probability distribution over moves (sums to 1.0)
    """
    policy_logits, value = model(board_tensor)
    # Set illegal moves to -inf before softmax
    policy_logits[~legal_moves_mask] = float('-inf')
    policy = F.softmax(policy_logits, dim=-1)
    return policy.squeeze(), value.squeeze()
```

### MCTS (`src/mcts/search.py`)

```python
class MCTSNode:
    """A node in the MCTS search tree.

    Attributes:
        parent: Parent node (None for root)
        action: The move (column) that led to this node
        prior: P(s,a) from neural network policy
        visit_count: N(s,a) — number of times this node was visited
        value_sum: W(s,a) — total value accumulated through this node
        children: dict mapping action → MCTSNode
        board: Connect4Board at this position
    """

class MCTS:
    """Monte Carlo Tree Search with neural network guidance.

    Args:
        model: Connect4Net for leaf evaluation
        config: MCTSConfig with num_simulations, c_puct, dirichlet params

    The search works as follows:
    1. SELECT: Traverse tree from root using PUCT formula
    2. EXPAND: Add leaf node, evaluate with neural network
    3. BACKUP: Propagate value up the path (flipping sign each level)
    4. After all simulations, return visit count distribution as policy
    """

    def search(self, board: Connect4Board) -> np.ndarray:
        """Run MCTS from the given position.

        Returns:
            (7,) array of visit count proportions for each column.
            Illegal moves will have 0 visits.
        """
```

**PUCT formula:**
```python
def _puct_score(self, parent: MCTSNode, child: MCTSNode) -> float:
    """Upper Confidence Bound for Trees.

    UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))

    Q(s,a) = W(s,a) / N(s,a)  (mean value of this action)
    P(s,a) = prior from neural network
    """
```

**Dirichlet noise** (exploration at root only):
```python
# Before search, add noise to root priors:
noise = np.random.dirichlet([alpha] * num_legal_moves)
root.prior = (1 - epsilon) * root.prior + epsilon * noise
# alpha = 1.0 (for ~7 legal moves in Connect 4)
# epsilon = 0.25
```

**Temperature-controlled move selection:**
```python
def select_move(visit_counts: np.ndarray, temperature: float) -> int:
    """Select a move from MCTS visit counts.

    If temperature ≈ 0: select the most-visited move (argmax)
    If temperature = 1: sample proportionally to visit counts
    In between: raise visit counts to power 1/temp, then sample
    """
    if temperature < 0.01:
        return np.argmax(visit_counts)
    adjusted = visit_counts ** (1.0 / temperature)
    probs = adjusted / adjusted.sum()
    return np.random.choice(len(probs), p=probs)
```

### Training Pipeline

#### Self-Play (`src/training/self_play.py`)

```python
def execute_self_play_game(model, mcts_config) -> list[TrainingSample]:
    """Play a complete game using MCTS, collecting training data.

    At each position:
    1. Run MCTS with current model
    2. Record (board_encoding, mcts_policy, current_player)
    3. Select move using temperature schedule
    4. Apply data augmentation (horizontal flip)

    After game ends:
    - Assign value labels: +1 for winner's positions, -1 for loser's, 0 for draw

    Returns:
        List of TrainingSample(state, policy, value) tuples
    """
```

**Temperature schedule:**
```python
# Moves 1-20: temperature = 1.0 (explore diverse openings)
# Moves 21+: temperature = 0.3 (exploit, play stronger moves)
```

#### Training (`src/training/trainer.py`)

```python
class Trainer:
    """Trains the neural network on self-play data.

    Loss = MSE(value_pred, value_target)
           + CrossEntropy(policy_pred, policy_target)
           + weight_decay * L2(parameters)

    Uses:
    - Adam optimizer, lr=1e-3
    - Batch size: 1024
    - Samples from replay buffer (FIFO, max 1M entries)
    """
```

#### Arena Evaluation (`src/training/arena.py`)

```python
def pit(model_new, model_old, num_games: int = 128) -> tuple[int, int, int]:
    """Play num_games between two models.

    Each model plays half as first player, half as second player.
    Uses low temperature (0.1) for near-deterministic play.

    Returns:
        (new_wins, old_wins, draws)
    """
```

#### Coach (`src/training/coach.py`)

```python
class Coach:
    """Orchestrates the full AlphaZero training loop.

    For each iteration:
    1. Generate self_play_games_per_iteration games
    2. Add data to replay buffer
    3. Train network for num_epochs epochs
    4. Evaluate new model vs current best in arena
    5. If new model wins > update_threshold: accept as new best
    6. Save checkpoint

    Logs: iteration number, losses, arena results, win rates
    """
```

### Configuration (`src/utils/config.py`)

Use dataclasses for all configuration:

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
    temperature_threshold: int = 20  # moves before switching temp
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
    game: GameConfig = field(default_factory=GameConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
```

Provide YAML loading: `Config.from_yaml("configs/tiny.yaml")`

---

## Development Workflow

### Iterative Build Order

Build in this exact order. **Each iteration must have passing tests before proceeding.**

#### Iteration 1: Game Engine
Build `src/game/board.py` and `tests/test_board.py`. Run tests. Verify bitboard correctness.

#### Iteration 2: Neural Network
Build `src/neural_net/model.py` and `tests/test_model.py`. Verify shapes, gradients, save/load.

#### Iteration 3: MCTS
Build `src/mcts/search.py` and `tests/test_mcts.py`. Test against known positions (must find winning move, must block opponent's win).

#### Iteration 4: Training Pipeline (First End-to-End)
Build `self_play.py`, `trainer.py`, `arena.py`, `coach.py`. Build `scripts/train.py`. Run with tiny config (2 blocks, 32 filters, 100 games, 50 sims). This produces v0.1 baseline. **Must beat random agent >90%.**

#### Iteration 5: Benchmarking
Build agents (`random_agent.py`, `minimax_agent.py`, `mcts_agent.py`). Build `scripts/evaluate.py`. Measure baseline against all opponents.

#### Iteration 6: Scale Training
Run full config on cloud GPU. Monitor loss curves. Produce production model.

#### Iteration 7: Export + Kaggle
Build `export/onnx_export.py` and `export/kaggle_agent.py`. Test with kaggle-environments locally. Submit.

#### Iteration 8: Web Interface
Build `web/` directory. Canvas board, ONNX Runtime Web inference, Web Worker for non-blocking AI.

### Running Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_board.py -v

# Train with tiny config (local testing)
python scripts/train.py --config configs/tiny.yaml

# Train with full config (GPU)
python scripts/train.py --config configs/full.yaml

# Evaluate model against agents
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --num-games 100

# Export to ONNX
python scripts/export_onnx.py --checkpoint checkpoints/best_model.pt --output model.onnx

# Package for Kaggle
python scripts/kaggle_submit.py --model model.onnx --output submission/
```

---

## Key Algorithm Details

### MCTS Simulation Budget

| Context | Simulations | Rationale |
|---|---|---|
| Self-play training | 600 | Quality training data |
| Arena evaluation | 200 | Faster evaluation, still strong |
| Kaggle competition | 100-400 | 2s time limit constrains this |
| Web (policy-only mode) | 0 | Raw network output, instant |
| Web (with MCTS) | 50-100 | Responsive feel in browser |

### Self-Play Data Format

Each training sample is a tuple:
```python
@dataclass
class TrainingSample:
    state: np.ndarray     # (3, 6, 7) board encoding
    policy: np.ndarray    # (7,) MCTS visit count distribution
    value: float          # +1 (win), -1 (loss), 0 (draw)
```

Data augmentation: for each sample, also add the horizontally-flipped version (flipped state, flipped policy). This doubles the dataset for free because Connect 4 is symmetric.

### Replay Buffer

FIFO (first-in, first-out) with configurable max size. When full, oldest samples are evicted. Samples are drawn uniformly at random for training mini-batches. This mixing of data from different iterations prevents overfitting to the current play style.

### Arena Evaluation Protocol

- New model plays `arena_num_games / 2` games as Player 1 and `arena_num_games / 2` as Player 2
- Both models use low temperature (0.1) for near-deterministic play
- If new model win% > `update_threshold` (55%): accept as new best
- This conservative threshold prevents regression from noise

---

## ONNX Export Details

```python
# Export PyTorch model to ONNX
dummy_input = torch.randn(1, 3, 6, 7)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=11,
    input_names=["board_state"],
    output_names=["policy_logits", "value"],
    dynamic_axes={"board_state": {0: "batch_size"}}
)

# Quantize for smaller size and faster inference
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("model.onnx", "model_quantized.onnx", weight_type=QuantType.QUInt8)
```

Target: ~6MB unquantized, ~2MB quantized.

---

## Kaggle Submission Template

The submission must be a single self-contained Python file:

```python
import numpy as np
import onnxruntime as ort
import math

# Global state — initialized on first call
_session = None
_initialized = False

def _initialize():
    global _session, _initialized
    _session = ort.InferenceSession('/kaggle/input/my-dataset/model.onnx')
    _initialized = True

def _encode_board(board_list, mark):
    """Convert Kaggle's flat board list to (1, 3, 6, 7) numpy array."""
    board = np.array(board_list).reshape(6, 7)
    state = np.zeros((1, 3, 6, 7), dtype=np.float32)
    state[0, 0] = (board == mark).astype(np.float32)
    state[0, 1] = (board == (3 - mark)).astype(np.float32)
    state[0, 2] = 1.0 if mark == 1 else 0.0  # turn indicator
    return state

def _get_legal_moves(board_list, cols=7):
    """Return list of columns that are not full."""
    return [c for c in range(cols) if board_list[c] == 0]

# Include simplified MCTS here or use raw policy

def my_agent(observation, configuration):
    global _initialized
    if not _initialized:
        _initialize()

    board = observation.board
    mark = observation.mark
    state = _encode_board(board, mark)

    # Run inference
    policy_logits, value = _session.run(None, {"board_state": state})
    policy_logits = policy_logits[0]

    # Mask illegal moves
    legal_moves = _get_legal_moves(board)
    masked = np.full(7, -1e9)
    for m in legal_moves:
        masked[m] = policy_logits[m]

    return int(np.argmax(masked))
```

---

## Web Interface Requirements

### Board Rendering (Canvas API)
- 7×6 grid with rounded-corner cells
- Smooth drop animation (piece falls with gravity easing)
- Hover preview showing which column the player is targeting
- Winning four highlighted after game ends
- Responsive: works on mobile and desktop

### AI Integration
- Use **Web Worker** (`ai_worker.js`) to prevent UI freezing during inference
- Load ONNX model once on worker initialization
- Worker receives board state, returns recommended move
- Show "thinking" indicator while AI computes

### Game Modes
- **Human vs AI** — player clicks column, AI responds
- **AI vs AI** — automated play with configurable speed
- **Difficulty levels** — vary MCTS simulations (0=instant/policy-only, 50=easy, 200=hard)

---

## Common Pitfalls to Avoid

1. **Slow Python MCTS:** The #1 risk. Python MCTS is often 10-100× too slow. Mitigations:
   - Use bitboard operations (not array-based board)
   - Batch neural network calls (collect multiple leaf positions, run one forward pass)
   - Consider Cython for the MCTS hot loop if needed (see `bhansconnect/fast-alphazero-general`)

2. **Value head collapse:** Network predicts ~0 for all positions. Check:
   - Value loss decreasing over training
   - Distribution of predicted values has variance
   - First-player and second-player data are balanced

3. **Catastrophic forgetting:** Agent loses early-game skill when training on late-game data. Fix: large replay buffer, shuffled mini-batches, data from multiple iterations.

4. **Reward shaping trap:** Do NOT add intermediate rewards (e.g., +0.1 for three-in-a-row). Use ONLY terminal rewards: +1 win, -1 loss, 0 draw. Intermediate rewards misalign incentives.

5. **Legal move masking bugs:** If illegal moves get non-zero probability, MCTS will select them. Always mask before softmax by setting logits to -inf.

6. **Perspective confusion:** The network always evaluates from the CURRENT player's perspective. When backpropagating MCTS values, flip the sign at each level. Getting this wrong produces an agent that plays to lose.

7. **Checkpoint corruption:** Always save both the model weights AND the optimizer state. Verify checkpoints can be loaded in a fresh process before long training runs.

---

## Performance Optimization Checklist

- [ ] Bitboard game engine (not array-based)
- [ ] Batch NN inference during MCTS (collect leaves, one forward pass)
- [ ] Mixed-precision training (torch.cuda.amp) on GPU
- [ ] Parallel self-play workers (multiprocessing)
- [ ] ONNX export with dynamic quantization
- [ ] NumPy vectorization where possible in data processing
- [ ] Profile before optimizing: use `cProfile` or `py-spy`

---

## Dependencies

### Python (`requirements.txt`)
```
torch>=2.0
numpy>=1.24
onnx>=1.14
onnxruntime>=1.15
pyyaml>=6.0
tqdm>=4.65
matplotlib>=3.7  # for training visualization
```

### Dev Dependencies
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
onnxruntime-web  # via npm or CDN
```

---

## Git Practices

- Commit after each iteration's tests pass
- Use conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`
- `.gitignore` checkpoints/, logs/, __pycache__/, *.onnx (except web/model.onnx)
- Tag releases: `v0.1-baseline`, `v0.2-trained`, `v1.0-deployed`
