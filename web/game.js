'use strict';

// ── Constants ────────────────────────────────────────────────────────────────
const COLS = 7;
const ROWS = 6;
const CELL_SIZE = 80;
const RADIUS = 34;
const PAD = 10;
const PREVIEW_H = 60;

const CANVAS_W = COLS * CELL_SIZE + 2 * PAD;
const CANVAS_H = ROWS * CELL_SIZE + 2 * PAD + PREVIEW_H;

const COLOR_BOARD   = '#1565c0';
const COLOR_EMPTY   = 'rgba(255,255,255,0.85)';
const COLOR_HUMAN   = '#e53935';
const COLOR_AI      = '#fdd835';
const COLOR_WIN_RING = '#00e676';
const COLOR_BG       = '#21262d';  // canvas background (outside board)

// ── DOM Refs ─────────────────────────────────────────────────────────────────
const canvas      = document.getElementById('gameCanvas');
const ctx         = canvas.getContext('2d');
const newGameBtn    = document.getElementById('newGameBtn');
const takeBackBtn   = document.getElementById('takeBackBtn');
const statusDot     = document.getElementById('statusDot');
const statusText    = document.getElementById('statusText');
const modelSizeSel  = document.getElementById('modelSize');
const lookaheadSel  = document.getElementById('lookahead');

// ── Game State ────────────────────────────────────────────────────────────────
// board[row][col], row 0 = BOTTOM. 0=empty, 1=human(red), 2=AI(yellow)
let board        = [];
let humanMark    = 1;   // 1 if human first, 2 if AI first
let aiMark       = 2;
let currentPlayer = 1;
let gameOver     = false;
let winCells     = null;  // array of {row,col} for winning 4 cells, or null
let hoveredCol   = -1;
let animating    = false;
let animPiece    = null;  // { col, row, color, currentY, targetY, startY, startTime }
let moveHistory  = [];    // [{col, row, mark}, ...] — one entry per placed piece
let aiThinking   = false; // true while waiting for worker response
let cancelAIMove = false; // flag to discard the next worker 'move' response

// ── Worker ────────────────────────────────────────────────────────────────────
let worker         = null;
let workerReady    = false;
let loadedModelUrl = '';   // URL currently loaded in the worker

// ── Canvas Setup ──────────────────────────────────────────────────────────────
function setupCanvas() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width  = CANVAS_W * dpr;
  canvas.height = CANVAS_H * dpr;
  canvas.style.width  = CANVAS_W + 'px';
  canvas.style.height = CANVAS_H + 'px';
  ctx.scale(dpr, dpr);
}

// ── Coordinate Helpers ────────────────────────────────────────────────────────
/** Return canvas X center for a given column. */
function cellX(col) { return PAD + col * CELL_SIZE + CELL_SIZE / 2; }

/** Return canvas Y center for a given board row.
 *  row 0 = BOTTOM → drawn at bottom of board;  row 5 = TOP → drawn at top of board.
 */
function cellY(row) {
  const canvasRow = (ROWS - 1) - row;   // flip: row 0 → canvasRow 5 (bottom of board)
  return PREVIEW_H + PAD + canvasRow * CELL_SIZE + CELL_SIZE / 2;
}

/** Return the canvas Y just above the board (start of drop animation). */
function aboveBoardY() { return PREVIEW_H / 2; }

/** Given a canvas X coordinate, return the column index (-1 if outside). */
function xToCol(x) {
  const col = Math.floor((x - PAD) / CELL_SIZE);
  return col >= 0 && col < COLS ? col : -1;
}

// ── Rendering ─────────────────────────────────────────────────────────────────
function pieceColor(mark) {
  if (mark === humanMark) return COLOR_HUMAN;
  if (mark === aiMark)    return COLOR_AI;
  return COLOR_EMPTY;
}

function drawCircle(x, y, r, fillColor, strokeColor, strokeWidth) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = fillColor;
  ctx.fill();
  if (strokeColor) {
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = strokeWidth || 3;
    ctx.stroke();
  }
}

function draw() {
  // Clear
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

  // Background (behind board)
  ctx.fillStyle = COLOR_BG;
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

  // Preview row: semi-transparent piece above hovered column
  if (hoveredCol >= 0 && !animating && !gameOver && currentPlayer === humanMark) {
    const px = cellX(hoveredCol);
    const py = aboveBoardY();
    ctx.globalAlpha = 0.55;
    drawCircle(px, py, RADIUS, COLOR_HUMAN, null, 0);
    ctx.globalAlpha = 1.0;
  }

  // Board rectangle (roundRect with fallback for older browsers)
  ctx.fillStyle = COLOR_BOARD;
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(PAD, PREVIEW_H + PAD, COLS * CELL_SIZE, ROWS * CELL_SIZE, 8);
  } else {
    ctx.rect(PAD, PREVIEW_H + PAD, COLS * CELL_SIZE, ROWS * CELL_SIZE);
  }
  ctx.fill();

  // Slot circles (white holes) + placed pieces
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const x = cellX(c);
      const y = cellY(r);
      const mark = board[r][c];

      // Skip the target cell of the animating piece (will be drawn during animation)
      if (animPiece && animPiece.col === c && animPiece.row === r) {
        drawCircle(x, y, RADIUS, COLOR_EMPTY, null, 0);
        continue;
      }

      if (mark === 0) {
        drawCircle(x, y, RADIUS, COLOR_EMPTY, null, 0);
      } else {
        drawCircle(x, y, RADIUS, pieceColor(mark), null, 0);
      }
    }
  }

  // Win highlight: glowing ring around winning 4
  if (winCells) {
    ctx.save();
    ctx.shadowColor = COLOR_WIN_RING;
    ctx.shadowBlur = 16;
    for (const { row, col } of winCells) {
      drawCircle(cellX(col), cellY(row), RADIUS, 'transparent', COLOR_WIN_RING, 5);
    }
    ctx.restore();
  }

  // Animating piece
  if (animPiece) {
    drawCircle(cellX(animPiece.col), animPiece.currentY, RADIUS, animPiece.color, null, 0);
  }
}

// ── Animation ─────────────────────────────────────────────────────────────────
const ANIM_DURATION = 350;  // ms

function startAnimation(col, row, mark) {
  animating = true;
  animPiece = {
    col,
    row,
    color: pieceColor(mark),
    currentY: aboveBoardY(),
    startY:   aboveBoardY(),
    targetY:  cellY(row),
    startTime: performance.now(),
    mark,
  };
  requestAnimationFrame(animFrame);
}

function animFrame(now) {
  if (!animPiece) return;
  const elapsed  = now - animPiece.startTime;
  const t        = Math.min(elapsed / ANIM_DURATION, 1.0);
  const progress = t * t;  // ease-in (gravity)
  animPiece.currentY = animPiece.startY + (animPiece.targetY - animPiece.startY) * progress;
  draw();

  if (t < 1.0) {
    requestAnimationFrame(animFrame);
  } else {
    // Animation complete
    const { col, row, mark } = animPiece;
    animPiece = null;
    animating = false;

    board[row][col] = mark;
    moveHistory.push({ col, row, mark });
    const wins = findWinCells(board, mark);
    if (wins) {
      winCells = wins;
      gameOver = true;
      const msg = mark === humanMark ? "You win! 🎉" : "AI wins!";
      setStatus(msg, mark === humanMark ? 'win' : 'ai');
    } else if (isDraw(board)) {
      gameOver = true;
      setStatus("Draw!", 'draw');
    } else {
      currentPlayer = 3 - mark;
      draw();
      if (currentPlayer === aiMark) {
        triggerAI();
      } else {
        setStatus("Your turn", 'human');
      }
    }
    draw();
  }
}

// ── Board Logic ───────────────────────────────────────────────────────────────
function emptyBoard() {
  return Array.from({ length: ROWS }, () => new Array(COLS).fill(0));
}

function getLegalCols(b) {
  const legal = [];
  for (let c = 0; c < COLS; c++) {
    if (b[ROWS - 1][c] === 0) legal.push(c);  // top row (row 5) empty → not full
  }
  return legal;
}

function dropRow(b, col) {
  // Return the row index where a piece lands in `col` (-1 if full).
  for (let row = 0; row < ROWS; row++) {
    if (b[row][col] === 0) return row;
  }
  return -1;
}

function isDraw(b) {
  return getLegalCols(b).length === 0;
}

function findWinCells(b, mark) {
  const directions = [[0,1],[1,0],[1,1],[1,-1]];
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLS; col++) {
      if (b[row][col] !== mark) continue;
      for (const [dr, dc] of directions) {
        const cells = [];
        for (let k = 0; k < 4; k++) {
          const r = row + dr * k;
          const c = col + dc * k;
          if (r < 0 || r >= ROWS || c < 0 || c >= COLS || b[r][c] !== mark) break;
          cells.push({ row: r, col: c });
        }
        if (cells.length === 4) return cells;
      }
    }
  }
  return null;
}

// ── Human Move ────────────────────────────────────────────────────────────────
function humanMove(col) {
  if (gameOver || animating || currentPlayer !== humanMark) return;
  const row = dropRow(board, col);
  if (row < 0) return;   // full column
  startAnimation(col, row, humanMark);
}

// ── AI Move ───────────────────────────────────────────────────────────────────
function triggerAI() {
  if (!workerReady) return;
  aiThinking = true;
  setStatus("AI is thinking...", 'loading');
  const flat = board.flat();  // 1D length 42, row 0 = BOTTOM
  const numSims = parseInt(lookaheadSel.value, 10);
  worker.postMessage({ type: 'getMove', board: flat, humanMark, numSims });
}

function aiMove(col) {
  aiThinking = false;
  if (gameOver) return;
  const row = dropRow(board, col);
  if (row < 0) return;
  startAnimation(col, row, aiMark);
}

// ── Take Back ─────────────────────────────────────────────────────────────────
function takeBack() {
  if (animating) return;

  // Require the human to have made at least one move
  if (!moveHistory.some(m => m.mark === humanMark)) return;

  if (aiThinking) {
    // AI is computing — cancel it and undo the human's last move
    cancelAIMove = true;
    aiThinking   = false;
    const { col, row } = moveHistory.pop();
    board[row][col] = 0;
  } else {
    // Undo AI's last move if it's on top, then undo human's last move
    if (moveHistory.length > 0 && moveHistory[moveHistory.length - 1].mark === aiMark) {
      const { col, row } = moveHistory.pop();
      board[row][col] = 0;
    }
    if (moveHistory.length > 0 && moveHistory[moveHistory.length - 1].mark === humanMark) {
      const { col, row } = moveHistory.pop();
      board[row][col] = 0;
    }
  }

  gameOver      = false;
  winCells      = null;
  currentPlayer = humanMark;
  setStatus("Your turn", 'human');
  draw();
}

// ── Status Bar ────────────────────────────────────────────────────────────────
const dotClasses = {
  idle:    'dot-idle',
  human:   'dot-human',
  ai:      'dot-ai',
  win:     'dot-win',
  draw:    'dot-draw',
  loading: 'dot-loading',
  error:   'dot-error',
};

function setStatus(msg, type) {
  statusText.textContent = msg;
  statusDot.className = 'dot ' + (dotClasses[type] || 'dot-idle');
}

// ── New Game ──────────────────────────────────────────────────────────────────
function startNewGame() {
  board         = emptyBoard();
  winCells      = null;
  gameOver      = false;
  animating     = false;
  animPiece     = null;
  hoveredCol    = -1;
  moveHistory   = [];
  aiThinking    = false;
  cancelAIMove  = false;

  humanMark     = document.getElementById('humanFirst').checked ? 1 : 2;
  aiMark        = 3 - humanMark;
  currentPlayer = 1;  // player 1 always goes first

  draw();

  const selectedModel = modelSizeSel.value;
  if (!workerReady || selectedModel !== loadedModelUrl) {
    // Worker not ready or model changed — (re)load
    workerReady    = false;
    loadedModelUrl = selectedModel;
    setStatus("Loading model...", 'loading');
    worker.postMessage({ type: 'loadModel', modelUrl: selectedModel });
    return;  // game starts when worker sends 'ready'
  }

  if (currentPlayer === humanMark) {
    setStatus("Your turn", 'human');
  } else {
    triggerAI();
  }
}

// ── Event Handlers ────────────────────────────────────────────────────────────
canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const col = xToCol(x);
  if (col !== hoveredCol) {
    hoveredCol = col;
    if (!animating) draw();
  }
});

canvas.addEventListener('mouseleave', () => {
  hoveredCol = -1;
  if (!animating) draw();
});

canvas.addEventListener('click', (e) => {
  if (gameOver || animating || currentPlayer !== humanMark) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const col = xToCol(x);
  if (col >= 0) humanMove(col);
});

newGameBtn.addEventListener('click', startNewGame);
takeBackBtn.addEventListener('click', takeBack);

// ── Worker Bootstrap ──────────────────────────────────────────────────────────
function initWorker() {
  worker = new Worker('ai_worker.js');
  worker.onmessage = (e) => {
    switch (e.data.type) {
      case 'ready':
        workerReady = true;
        if (!gameOver && currentPlayer === aiMark) {
          triggerAI();
        } else if (!gameOver) {
          setStatus("Your turn", 'human');
        }
        break;
      case 'move':
        if (cancelAIMove) { cancelAIMove = false; aiThinking = false; break; }
        aiMove(e.data.col);
        break;
      case 'error':
        setStatus("Error: " + e.data.message, 'error');
        break;
    }
  };
  worker.onerror = (err) => {
    setStatus("Worker error: " + err.message, 'error');
  };
  loadedModelUrl = modelSizeSel.value;
  worker.postMessage({ type: 'init', modelUrl: loadedModelUrl });
}

// ── Init ──────────────────────────────────────────────────────────────────────
setupCanvas();
board = emptyBoard();
draw();
setStatus("Loading AI model...", 'loading');
initWorker();
