'use strict';

// ── ONNX Runtime Web ──────────────────────────────────────────────────────────
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');

let session = null;

// ── Constants ─────────────────────────────────────────────────────────────────
const ROWS    = 6;
const COLS    = 7;
const C_PUCT  = 2.0;

// ── Message Handler ───────────────────────────────────────────────────────────
self.onmessage = async (e) => {
  try {
    if (e.data.type === 'init') {
      await initModel(e.data.modelUrl);
      self.postMessage({ type: 'ready' });
    } else if (e.data.type === 'getMove') {
      const col = await getMove(e.data);
      self.postMessage({ type: 'move', col });
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: err.message });
  }
};

async function initModel(modelUrl) {
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
  session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ['wasm'],
  });
}

// ── Board Logic ───────────────────────────────────────────────────────────────
// board: 1D Float32Array / Array, length 42.
// Index: board[row * 7 + col],  row 0 = BOTTOM (gravity side).
// mark: 1 = player-1, 2 = player-2.

function getLegalMoves(board) {
  const legal = [];
  for (let col = 0; col < COLS; col++) {
    if (board[(ROWS - 1) * COLS + col] === 0) legal.push(col);  // top row empty
  }
  return legal;
}

/** Return a fresh copy of board with `mark` dropped in `col`.  Does not mutate. */
function makeMove(board, col, mark) {
  const next = board.slice();
  for (let row = 0; row < ROWS; row++) {
    if (next[row * COLS + col] === 0) {
      next[row * COLS + col] = mark;
      break;
    }
  }
  return next;
}

function checkWin(board, mark) {
  const directions = [[0,1],[1,0],[1,1],[1,-1]];
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLS; col++) {
      if (board[row * COLS + col] !== mark) continue;
      for (const [dr, dc] of directions) {
        let cnt = 0;
        for (let k = 0; k < 4; k++) {
          const r = row + dr * k;
          const c = col + dc * k;
          if (r < 0 || r >= ROWS || c < 0 || c >= COLS) break;
          if (board[r * COLS + c] === mark) cnt++; else break;
        }
        if (cnt === 4) return true;
      }
    }
  }
  return false;
}

function isDraw(board) {
  return getLegalMoves(board).length === 0;
}

function isTerminal(board) {
  return checkWin(board, 1) || checkWin(board, 2) || isDraw(board);
}

// ── Board Encoding ────────────────────────────────────────────────────────────
/**
 * Encode board into 3-plane tensor (3×6×7) matching Python board.py:
 *   plane 0: current-player pieces
 *   plane 1: opponent pieces
 *   plane 2: P1-to-move flag (all 1.0 if currentPlayer===1, else 0.0)
 *
 * Row 0 = BOTTOM in both Python and this JS — no flip needed.
 */
function encodeBoard(board, currentPlayer) {
  const state = new Float32Array(3 * ROWS * COLS);
  const opponent = 3 - currentPlayer;
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLS; col++) {
      const idx = row * COLS + col;
      const cell = board[idx];
      if (cell === currentPlayer) {
        state[idx] = 1.0;           // plane 0
      } else if (cell === opponent) {
        state[ROWS * COLS + idx] = 1.0;  // plane 1
      }
    }
  }
  if (currentPlayer === 1) {
    state.fill(1.0, 2 * ROWS * COLS, 3 * ROWS * COLS);  // plane 2
  }
  return state;
}

// ── Neural Network ────────────────────────────────────────────────────────────
async function runNN(boardState) {
  const tensor = new ort.Tensor('float32', boardState, [1, 3, ROWS, COLS]);
  const output = await session.run({ board_state: tensor });
  const logits = Array.from(output.policy_logits.data);  // length 7
  const value  = output.value.data[0];                    // float in [-1,1]
  return [logits, value];
}

function maskedSoftmax(logits, legalMoves) {
  const masked = new Array(COLS).fill(-Infinity);
  for (const col of legalMoves) masked[col] = logits[col];

  let maxVal = -Infinity;
  for (const v of masked) { if (isFinite(v) && v > maxVal) maxVal = v; }

  const exp = masked.map(x => isFinite(x) ? Math.exp(x - maxVal) : 0.0);
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(x => x / sum);
}

// ── MCTS Node ─────────────────────────────────────────────────────────────────
class MCTSNode {
  constructor(parent, action, prior) {
    this.parent     = parent;
    this.action     = action;
    this.prior      = prior;
    this.visitCount = 0;
    this.valueSum   = 0.0;
    this.children   = new Map();  // col → MCTSNode
  }

  get qValue() {
    return this.visitCount > 0 ? this.valueSum / this.visitCount : 0.0;
  }

  get isLeaf() { return this.children.size === 0; }
}

function puctScore(parent, child) {
  const exploration = C_PUCT * child.prior * Math.sqrt(parent.visitCount) / (1 + child.visitCount);
  return -child.qValue + exploration;  // negate: child's perspective → parent's perspective
}

// ── MCTS Search ───────────────────────────────────────────────────────────────
async function mcts(board, aiMark, numSims) {
  const legal = getLegalMoves(board);
  if (legal.length === 0) return -1;

  // Policy-only (no simulations): argmax over policy priors
  if (numSims === 0) {
    const state = encodeBoard(board, aiMark);
    const [logits] = await runNN(state);
    const probs = maskedSoftmax(logits, legal);
    let bestCol = legal[0], bestProb = probs[legal[0]];
    for (const col of legal) {
      if (probs[col] > bestProb) { bestProb = probs[col]; bestCol = col; }
    }
    return bestCol;
  }

  // Initialize root and expand it
  const root = new MCTSNode(null, -1, 1.0);
  const state0 = encodeBoard(board, aiMark);
  const [logits0] = await runNN(state0);
  const probs0 = maskedSoftmax(logits0, legal);
  for (const col of legal) {
    root.children.set(col, new MCTSNode(root, col, probs0[col]));
  }

  for (let sim = 0; sim < numSims; sim++) {
    let node      = root;
    let simBoard  = board.slice();
    let simPlayer = aiMark;

    // ── SELECT ──────────────────────────────────────────────────────────────
    while (!node.isLeaf && !isTerminal(simBoard)) {
      let bestCol = -1, bestScore = -Infinity;
      for (const [col, child] of node.children) {
        const score = puctScore(node, child);
        if (score > bestScore) { bestScore = score; bestCol = col; }
      }
      node      = node.children.get(bestCol);
      simBoard  = makeMove(simBoard, bestCol, simPlayer);
      simPlayer = 3 - simPlayer;
    }

    // ── EVALUATE ─────────────────────────────────────────────────────────────
    let value;
    if (isTerminal(simBoard)) {
      // lastPlayer is the one who just moved (simPlayer has already been flipped)
      const lastPlayer = 3 - simPlayer;
      if (checkWin(simBoard, lastPlayer)) {
        // lastPlayer won → from simPlayer's (= current node's) perspective: loss
        value = -1.0;
      } else {
        value = 0.0;  // draw
      }
    } else {
      // Expand leaf
      const simState  = encodeBoard(simBoard, simPlayer);
      const [simLogits, val] = await runNN(simState);
      value = val;
      const simLegal = getLegalMoves(simBoard);
      const simProbs = maskedSoftmax(simLogits, simLegal);
      for (const col of simLegal) {
        node.children.set(col, new MCTSNode(node, col, simProbs[col]));
      }
    }

    // ── BACKUP ───────────────────────────────────────────────────────────────
    let backNode = node;
    while (backNode !== null) {
      backNode.visitCount += 1;
      backNode.valueSum   += value;
      value    = -value;
      backNode = backNode.parent;
    }
  }

  // Pick most-visited child of root
  let bestCol = -1, bestVisits = -1;
  for (const [col, child] of root.children) {
    if (child.visitCount > bestVisits) {
      bestVisits = child.visitCount;
      bestCol    = col;
    }
  }
  return bestCol;
}

// ── Entry Point ───────────────────────────────────────────────────────────────
async function getMove({ board, humanMark, numSims }) {
  const aiMark = 3 - humanMark;
  return await mcts(board, aiMark, numSims);
}
