"""Self-contained Kaggle Connect-X agent — pure numpy, zero external deps.

Only stdlib + numpy. No onnxruntime, no torch, no scipy.

Weights are loaded from a .npz file bundled in the submission archive.
The path is set by kaggle_submit.py --tar at packaging time.

Kaggle extracts the tar.gz to /kaggle_simulations/agent/ — main.py and
weights.npz both land there, so the default path works as-is.

Board encoding note (CRITICAL):
  Kaggle: board_flat[kaggle_row * 7 + col], row 0 = TOP
  Network: state[plane, network_row, col], row 0 = BOTTOM (gravity)
  Conversion: network_row = (ROWS - 1) - kaggle_row
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

# -------------------------------------------------------------------------
# Sentinels — replaced by kaggle_submit.py --tar at packaging time
# -------------------------------------------------------------------------
_CWD = '/kaggle_simulations/agent/'
if os.path.exists(_CWD):
    sys.path.append(_CWD)
else:
    _CWD = ''

_WEIGHTS_PATH: str = _CWD + "weights.npz"
_TIME_BUDGET_SECS: float = 1.9  # sentinel

# -------------------------------------------------------------------------
# Section 1: Board logic
# -------------------------------------------------------------------------

_ROWS: int = 6
_COLS: int = 7
_WIN: int = 4


def _get_legal_moves(board_flat: list[int]) -> list[int]:
    return [c for c in range(_COLS) if board_flat[c] == 0]


def _make_move(board_flat: list[int], col: int, mark: int) -> list[int]:
    board = list(board_flat)
    for row in range(_ROWS - 1, -1, -1):
        if board[row * _COLS + col] == 0:
            board[row * _COLS + col] = mark
            return board
    raise ValueError(f"Column {col} is full")


def _check_win(board_flat: list[int], mark: int) -> bool:
    board = [board_flat[r * _COLS : (r + 1) * _COLS] for r in range(_ROWS)]
    for r in range(_ROWS):
        for c in range(_COLS - _WIN + 1):
            if all(board[r][c + i] == mark for i in range(_WIN)):
                return True
    for r in range(_ROWS - _WIN + 1):
        for c in range(_COLS):
            if all(board[r + i][c] == mark for i in range(_WIN)):
                return True
    for r in range(_ROWS - _WIN + 1):
        for c in range(_COLS - _WIN + 1):
            if all(board[r + i][c + i] == mark for i in range(_WIN)):
                return True
    for r in range(_WIN - 1, _ROWS):
        for c in range(_COLS - _WIN + 1):
            if all(board[r - i][c + i] == mark for i in range(_WIN)):
                return True
    return False


def _is_terminal(board_flat: list[int]) -> bool:
    return (
        _check_win(board_flat, 1)
        or _check_win(board_flat, 2)
        or len(_get_legal_moves(board_flat)) == 0
    )


def _get_result(board_flat: list[int], mark: int) -> float:
    if _check_win(board_flat, mark):
        return 1.0
    if _check_win(board_flat, 3 - mark):
        return -1.0
    return 0.0


def _encode_board(board_flat: list[int], current_mark: int) -> np.ndarray:
    """Encode board as (1, 3, 6, 7) float32. Row flip: Kaggle 0=top, network 0=bottom."""
    state = np.zeros((1, 3, _ROWS, _COLS), dtype=np.float32)
    for kaggle_row in range(_ROWS):
        network_row = (_ROWS - 1) - kaggle_row
        for col in range(_COLS):
            cell = board_flat[kaggle_row * _COLS + col]
            if cell == current_mark:
                state[0, 0, network_row, col] = 1.0
            elif cell == (3 - current_mark):
                state[0, 1, network_row, col] = 1.0
    state[0, 2, :, :] = 1.0 if current_mark == 1 else 0.0
    return state


# -------------------------------------------------------------------------
# Section 2: Numpy ResNet forward pass
# -------------------------------------------------------------------------

_weights: dict | None = None


def _load_weights() -> dict:
    global _weights
    if _weights is None:
        npz = np.load(_WEIGHTS_PATH)
        _weights = {k: npz[k] for k in npz.files}
    return _weights


def _conv2d(x: np.ndarray, w: np.ndarray, pad: int = 1) -> np.ndarray:
    """2D convolution via im2col + matmul. No bias (BN follows).

    Args:
        x: (1, C, H, W) float32
        w: (F, C, kH, kW) float32
        pad: zero-padding on each spatial side

    Returns:
        (1, F, H_out, W_out) float32
    """
    if pad > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    _, C, H, W = x.shape
    F, _, kH, kW = w.shape
    H_out = H - kH + 1
    W_out = W - kW + 1

    col = np.empty((H_out * W_out, C * kH * kW), dtype=np.float32)
    idx = 0
    for h in range(H_out):
        for wi in range(W_out):
            col[idx] = x[0, :, h : h + kH, wi : wi + kW].ravel()
            idx += 1

    out = col @ w.reshape(F, -1).T
    return out.T.reshape(1, F, H_out, W_out)


def _bn(x, gamma, beta, mean, var, eps=1e-5):
    """Batch norm (eval mode) for (1, C, H, W) tensors."""
    return (
        gamma[None, :, None, None]
        * (x - mean[None, :, None, None])
        / np.sqrt(var[None, :, None, None] + eps)
        + beta[None, :, None, None]
    )


def _predict(state: np.ndarray) -> tuple[np.ndarray, float]:
    """Run ResNet forward pass on (1, 3, 6, 7) board encoding.

    Returns:
        (policy_logits, value): (7,) float32 logits and scalar in [-1, 1]
    """
    w = _load_weights()
    num_blocks = int(w["num_blocks"])

    # Stem
    x = _conv2d(state, w["stem.0.weight"], pad=1)
    x = _bn(x, w["stem.1.weight"], w["stem.1.bias"],
            w["stem.1.running_mean"], w["stem.1.running_var"])
    x = np.maximum(x, 0)

    # Residual tower
    for i in range(num_blocks):
        residual = x
        out = _conv2d(x, w[f"tower.{i}.conv1.weight"], pad=1)
        out = _bn(out, w[f"tower.{i}.bn1.weight"], w[f"tower.{i}.bn1.bias"],
                  w[f"tower.{i}.bn1.running_mean"], w[f"tower.{i}.bn1.running_var"])
        out = np.maximum(out, 0)
        out = _conv2d(out, w[f"tower.{i}.conv2.weight"], pad=1)
        out = _bn(out, w[f"tower.{i}.bn2.weight"], w[f"tower.{i}.bn2.bias"],
                  w[f"tower.{i}.bn2.running_mean"], w[f"tower.{i}.bn2.running_var"])
        x = np.maximum(out + residual, 0)

    # Policy head: 1×1 conv → BN → ReLU → flatten → linear
    p = _conv2d(x, w["policy_head.0.weight"], pad=0)
    p = _bn(p, w["policy_head.1.weight"], w["policy_head.1.bias"],
            w["policy_head.1.running_mean"], w["policy_head.1.running_var"])
    p = np.maximum(p, 0).reshape(1, -1)
    policy_logits = (p @ w["policy_head.4.weight"].T + w["policy_head.4.bias"])[0]

    # Value head: 1×1 conv → BN → ReLU → flatten → linear → ReLU → linear → tanh
    v = _conv2d(x, w["value_head.0.weight"], pad=0)
    v = _bn(v, w["value_head.1.weight"], w["value_head.1.bias"],
            w["value_head.1.running_mean"], w["value_head.1.running_var"])
    v = np.maximum(v, 0).reshape(1, -1)
    v = np.maximum(v @ w["value_head.4.weight"].T + w["value_head.4.bias"], 0)
    v = v @ w["value_head.6.weight"].T + w["value_head.6.bias"]
    value = float(np.tanh(v[0, 0]))

    return policy_logits, value


# -------------------------------------------------------------------------
# Section 3: Inline MCTS
# -------------------------------------------------------------------------

_C_PUCT: float = 2.0


class _MCTSNode:
    __slots__ = ["board_flat", "mark", "parent", "action", "prior",
                 "visit_count", "value_sum", "children"]

    def __init__(self, board_flat, mark, parent=None, action=None, prior=0.0):
        self.board_flat = board_flat
        self.mark = mark
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: dict = {}

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def _puct_score(parent: _MCTSNode, child: _MCTSNode) -> float:
    exploration = (
        _C_PUCT * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    )
    return -child.q_value + exploration


def _expand(node: _MCTSNode) -> float:
    if _is_terminal(node.board_flat):
        return _get_result(node.board_flat, node.mark)
    if node.children:
        return 0.0
    state = _encode_board(node.board_flat, node.mark)
    policy_logits, value = _predict(state)
    legal = _get_legal_moves(node.board_flat)
    mask = np.full(_COLS, -1e9, dtype=np.float32)
    for c in legal:
        mask[c] = policy_logits[c]
    mask -= mask.max()
    exp = np.exp(mask)
    probs = exp / exp.sum()
    next_mark = 3 - node.mark
    for col in legal:
        node.children[col] = _MCTSNode(
            board_flat=_make_move(node.board_flat, col, node.mark),
            mark=next_mark, parent=node, action=col, prior=float(probs[col]),
        )
    return value


def _backup(node: _MCTSNode, value: float) -> None:
    current = node
    v = value
    while current is not None:
        current.visit_count += 1
        current.value_sum += v
        v = -v
        current = current.parent


def _mcts_search(board_flat: list[int], mark: int, time_budget_secs: float) -> tuple[int, float, int]:
    import time
    root = _MCTSNode(board_flat=list(board_flat), mark=mark)
    _expand(root)
    sims = 0
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < time_budget_secs:
        node = root
        while node.children and not _is_terminal(node.board_flat):
            best_score = -float("inf")
            best_child = None
            for child in node.children.values():
                score = _puct_score(node, child)
                if score > best_score:
                    best_score = score
                    best_child = child
            node = best_child  # type: ignore[assignment]
        _backup(node, _expand(node))
        sims += 1
    best_move = max(root.children, key=lambda c: root.children[c].visit_count)
    return best_move, root.q_value, sims


# -------------------------------------------------------------------------
# Section 4: Agent entry point
# -------------------------------------------------------------------------

def my_agent(observation, configuration) -> int:
    """Kaggle Connect-X agent — pure numpy ResNet + MCTS, no onnxruntime."""
    best_move, value, sims = _mcts_search(
        observation.board, observation.mark, _TIME_BUDGET_SECS
    )
    print(f"[agent] sims={sims}  value={value:+.3f}  move={best_move}")
    return best_move
