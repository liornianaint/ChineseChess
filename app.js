import {
  initialBoard,
  generateLegalMoves,
  makeMove,
  gameStatus,
  searchBestMove,
  searchBestMoveFromMoves,
  defaultWeights,
  isRed,
  isBlack,
  opposite,
  findCheckers,
  pieceType,
  indexToCoord,
  coordToIndex,
  positionKey,
} from './engine.js';

const DEFAULT_BACKEND_URL = 'http://127.0.0.1:8001';
const PYTHON_TRAIN_KEY = 'xiangqi_python_train_cfg_v4';
const TOTAL_GAMES_KEY = 'xiangqi_total_games_v1';

const DEFAULT_PYTHON_TRAIN = {
  iterations: 600,
  gamesPerIter: 12,
  simulations: 480,
  lr: 0.002,
  batchSize: 128,
  epochs: 2,
  bufferSize: 15000,
  advanced: false,
  preset: 'balanced',
};

const PYTHON_TRAIN_PRESETS = {
  balanced: { lr: 0.002, batchSize: 128, epochs: 2, bufferSize: 15000 },
  speed: { lr: 0.003, batchSize: 96, epochs: 1, bufferSize: 10000 },
  strong: { lr: 0.001, batchSize: 192, epochs: 3, bufferSize: 20000 },
};
const REPETITION_LIMIT = 3;
const LONG_CHECK_LIMIT = 6;

const PIECE_LABELS = {
  K: '帅',
  A: '仕',
  B: '相',
  N: '马',
  R: '车',
  C: '炮',
  P: '兵',
  k: '将',
  a: '士',
  b: '象',
  n: '马',
  r: '车',
  c: '炮',
  p: '卒',
};

const PIECE_LIST = ['K', 'A', 'B', 'N', 'R', 'C', 'P'];
const AI_TIME_LIMITS = {
  2: 320,
  3: 520,
  4: 760,
  5: 1050,
  6: 1400,
};

const MAX_SEARCH_THREADS = 12;
const DEFAULT_SEARCH_THREADS = Math.max(2, Math.min(MAX_SEARCH_THREADS, navigator.hardwareConcurrency || 4));
const MIN_MULTI_DEPTH = 3;

const PYTHON_SIM_MAP = {
  2: 300,
  3: 600,
  4: 900,
  5: 1200,
  6: 1500,
};

const boardEl = document.getElementById('board');
const statusEl = document.getElementById('status');

const aiRedEl = document.getElementById('aiRed');
const aiBlackEl = document.getElementById('aiBlack');
const depthEl = document.getElementById('depth');
const newGameEl = document.getElementById('newGame');
const undoEl = document.getElementById('undoMove');
const aiMoveEl = document.getElementById('aiMove');
const autoPlayEl = document.getElementById('autoPlay');
const openingBookEl = document.getElementById('openingBook');
const backendStatusEl = document.getElementById('backendStatus');

const pyTrainItersEl = document.getElementById('pyTrainIters');
const pyTrainGamesEl = document.getElementById('pyTrainGames');
const pyTrainSimsEl = document.getElementById('pyTrainSims');
const pyTrainAdvancedEl = document.getElementById('pyTrainAdvanced');
const pyTrainAdvancedPanelEl = document.getElementById('pyTrainAdvancedPanel');
const pyTrainPresetEl = document.getElementById('pyTrainPreset');
const pyTrainLrEl = document.getElementById('pyTrainLR');
const pyTrainBatchEl = document.getElementById('pyTrainBatch');
const pyTrainEpochsEl = document.getElementById('pyTrainEpochs');
const pyTrainBufferEl = document.getElementById('pyTrainBuffer');
const pyTrainStartEl = document.getElementById('pyTrainStart');
const pyTrainStopEl = document.getElementById('pyTrainStop');
const pyTrainLogEl = document.getElementById('pyTrainLog');
const pyTrainLogToggleEl = document.getElementById('pyTrainLogToggle');
const pyTrainLogMetaEl = document.getElementById('pyTrainLogMeta');
const pyTrainIterProgressEl = document.getElementById('pyTrainIterProgress');

const state = {
  board: initialBoard(),
  sideToMove: 'r',
  selected: null,
  legalMoves: [],
  history: [],
  lastMove: null,
  lastMoveByAI: false,
  aiSide: { r: false, b: true },
  autoPlay: true,
  aiThinking: false,
  gameOver: false,
  weights: defaultWeights(),
  positionCounts: new Map(),
  checkStreak: { side: null, count: 0 },
  backendOnline: false,
  backendDevice: '',
  backendModelLabel: '',
  backendModelReady: false,
  backendTotalGames: null,
  backendLastCheck: 0,
  lastBookMove: false,
  useOpeningBook: true,
  useMultiThread: true,
  searchThreads: DEFAULT_SEARCH_THREADS,
};

let traceEl = null;
let audioContext = null;

const searchWorkers = [];
let searchRequestId = 0;
let pythonTrainTimer = null;
let pythonTrainConfig = loadPythonTrainConfig();
let pythonTrainLogExpanded = false;
let pythonTrainLogLines = [];
let pythonTrainStartTime = null;
let pythonTrainProgress = { current: 0, total: 0 };
let pythonTrainIterProgress = { current: 0, total: 0, pct: 0 };
let pythonTrainRunning = false;
let pythonTrainStartPending = false;
let pythonTrainStartRequestedAt = 0;
let pythonTrainGamesDone = 0;
let pythonTrainBaseTotalGames = loadTotalGames();
let pythonTrainPostStopChecks = 0;
let pythonTrainSyncPending = false;
let pythonTrainSyncAttempts = 0;

function loadTotalGames() {
  const raw = localStorage.getItem(TOTAL_GAMES_KEY);
  if (!raw) return null;
  const num = Number(raw);
  return Number.isFinite(num) ? num : null;
}

function saveTotalGames(value) {
  if (!Number.isFinite(value)) return;
  localStorage.setItem(TOTAL_GAMES_KEY, String(value));
}

function clearTotalGames() {
  localStorage.removeItem(TOTAL_GAMES_KEY);
}

function loadPythonTrainConfig() {
  localStorage.removeItem('xiangqi_python_train_cfg_v1');
  localStorage.removeItem('xiangqi_python_train_cfg_v2');
  localStorage.removeItem('xiangqi_python_train_cfg_v3');
  const raw = localStorage.getItem(PYTHON_TRAIN_KEY);
  if (!raw) return { ...DEFAULT_PYTHON_TRAIN };
  try {
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_PYTHON_TRAIN, ...parsed };
  } catch (err) {
    return { ...DEFAULT_PYTHON_TRAIN };
  }
}

function savePythonTrainConfig(next) {
  pythonTrainConfig = { ...pythonTrainConfig, ...next };
  localStorage.setItem(PYTHON_TRAIN_KEY, JSON.stringify(pythonTrainConfig));
}

function getBackendBaseUrl() {
  return DEFAULT_BACKEND_URL.replace(/\/$/, '');
}

function clampInt(value, min, max, fallback) {
  const num = parseInt(value, 10);
  if (!Number.isFinite(num)) return fallback;
  return Math.max(min, Math.min(max, num));
}

function clampFloat(value, min, max, fallback) {
  const num = parseFloat(value);
  if (!Number.isFinite(num)) return fallback;
  return Math.max(min, Math.min(max, num));
}

function applyPythonTrainPreset(key) {
  const preset = PYTHON_TRAIN_PRESETS[key];
  if (!preset) return;
  if (pyTrainLrEl) pyTrainLrEl.value = preset.lr;
  if (pyTrainBatchEl) pyTrainBatchEl.value = preset.batchSize;
  if (pyTrainEpochsEl) pyTrainEpochsEl.value = preset.epochs;
  if (pyTrainBufferEl) pyTrainBufferEl.value = preset.bufferSize;
  if (pyTrainPresetEl) pyTrainPresetEl.value = key;
  savePythonTrainConfig({ ...preset, preset: key });
}

function setPresetCustom() {
  if (pyTrainPresetEl && pyTrainPresetEl.value !== 'custom') {
    pyTrainPresetEl.value = 'custom';
    savePythonTrainConfig({ preset: 'custom' });
  }
}

function pythonSimsForDepth(depth) {
  return PYTHON_SIM_MAP[depth] || 240;
}

function setStatus(text) {
  statusEl.textContent = text;
  if (typeof scheduleTrainCardSync === 'function') {
    scheduleTrainCardSync();
  }
}

function findKingIndexForSide(board, side) {
  return board.findIndex((piece) => {
    if (!piece || pieceType(piece) !== 'K') return false;
    return side === 'r' ? isRed(piece) : isBlack(piece);
  });
}

function horseLegIndex(horseIdx, kingIdx) {
  const { x: hx, y: hy } = indexToCoord(horseIdx);
  const { x: kx, y: ky } = indexToCoord(kingIdx);
  const dx = kx - hx;
  const dy = ky - hy;
  if (Math.abs(dx) === 2 && Math.abs(dy) === 1) {
    return coordToIndex(hx + (dx > 0 ? 1 : -1), hy);
  }
  if (Math.abs(dx) === 1 && Math.abs(dy) === 2) {
    return coordToIndex(hx, hy + (dy > 0 ? 1 : -1));
  }
  return -1;
}

function getCheckers(board, side) {
  const raw = findCheckers(board, side);
  if (!raw.length) return raw;
  const kingIdx = findKingIndexForSide(board, side);
  if (kingIdx === -1) return raw;
  return raw.filter((idx) => {
    const piece = board[idx];
    if (!piece || pieceType(piece) !== 'N') return true;
    const legIdx = horseLegIndex(idx, kingIdx);
    if (legIdx === -1) return true;
    return !board[legIdx];
  });
}

function setTurnStatus() {
  const base = state.sideToMove === 'r' ? '轮到红方' : '轮到黑方';
  if (getCheckers(state.board, state.sideToMove).length > 0) {
    setStatus(`${base}（被将军）`);
  } else {
    setStatus(base);
  }
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m ${String(sec).padStart(2, '0')}s`;
  return `${m}m ${String(sec).padStart(2, '0')}s`;
}

function playCheckmateSound() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    if (!audioContext) audioContext = new AudioCtx();
    const ctx = audioContext;
    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(440, now);
    osc.frequency.exponentialRampToValueAtTime(880, now + 0.25);
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.18, now + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.6);
    osc.connect(gain).connect(ctx.destination);
    osc.start(now);
    osc.stop(now + 0.65);
  } catch (err) {
    // ignore audio errors
  }
}

function createBoard() {
  boardEl.innerHTML = '';
  traceEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  traceEl.setAttribute('id', 'moveTrace');
  traceEl.setAttribute('aria-hidden', 'true');
  traceEl.classList.add('move-trace');
  boardEl.appendChild(traceEl);

  const river = document.createElement('div');
  river.className = 'river';
  river.innerHTML = '<span>楚河</span><span>汉界</span>';
  boardEl.appendChild(river);

  for (let y = 0; y < 10; y += 1) {
    for (let x = 0; x < 9; x += 1) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.dataset.idx = String(y * 9 + x);
      boardEl.appendChild(cell);
    }
  }
  boardEl.addEventListener('click', onBoardClick);
}

function renderTrace() {
  if (!traceEl) return;
  traceEl.innerHTML = '';
  traceEl.classList.remove('trace-animate');
  if (!state.lastMove || !state.lastMoveByAI) return;

  const fromCell = boardEl.querySelector(`.cell[data-idx="${state.lastMove.from}"]`);
  const toCell = boardEl.querySelector(`.cell[data-idx="${state.lastMove.to}"]`);
  if (!fromCell || !toCell) return;

  const boardRect = boardEl.getBoundingClientRect();
  if (boardRect.width === 0 || boardRect.height === 0) return;

  const fromRect = fromCell.getBoundingClientRect();
  const toRect = toCell.getBoundingClientRect();
  const fx = fromRect.left - boardRect.left + fromRect.width / 2;
  const fy = fromRect.top - boardRect.top + fromRect.height / 2;
  const tx = toRect.left - boardRect.left + toRect.width / 2;
  const ty = toRect.top - boardRect.top + toRect.height / 2;

  const color = isRed(state.lastMove.piece) ? '#b23a1b' : '#1f1f1f';
  const stroke = Math.max(2, Math.min(4, boardRect.width / 180));
  const markerSize = stroke * 2.8;

  traceEl.setAttribute('width', boardRect.width);
  traceEl.setAttribute('height', boardRect.height);
  traceEl.setAttribute('viewBox', `0 0 ${boardRect.width} ${boardRect.height}`);

  traceEl.innerHTML = `    <defs>
      <marker id="arrowhead" markerWidth="${markerSize}" markerHeight="${markerSize}"
        refX="${markerSize * 0.6}" refY="${markerSize / 2}" orient="auto" markerUnits="userSpaceOnUse">
        <path d="M0,0 L${markerSize},${markerSize / 2} L0,${markerSize} Z" fill="${color}" />
      </marker>
    </defs>
    <line class="trace-line" x1="${fx}" y1="${fy}" x2="${tx}" y2="${ty}" stroke="${color}" stroke-width="${stroke}" stroke-linecap="round" marker-end="url(#arrowhead)" />
    <circle class="trace-dot trace-dot-start" cx="${fx}" cy="${fy}" r="${stroke * 1.1}" fill="${color}" opacity="0.6" />
    <circle class="trace-dot trace-dot-end" cx="${tx}" cy="${ty}" r="${stroke * 1.4}" fill="${color}" opacity="0.85" />
  `;

  traceEl.getBoundingClientRect();
  traceEl.classList.add('trace-animate');
}

function getSearchWorker(index) {
  if (!searchWorkers[index]) {
    searchWorkers[index] = new Worker('search-worker.js', { type: 'module' });
  }
  return searchWorkers[index];
}

function runWorkerSearch(worker, payload, timeoutMs) {
  return new Promise((resolve) => {
    const id = ++searchRequestId;
    let settled = false;
    const timer = timeoutMs
      ? setTimeout(() => {
          if (settled) return;
          settled = true;
          cleanup();
          resolve({ timeout: true });
        }, timeoutMs + 120)
      : null;

    const cleanup = () => {
      if (timer) clearTimeout(timer);
      worker.removeEventListener('message', onMessage);
      worker.removeEventListener('error', onError);
    };

    const onMessage = (event) => {
      if (!event.data || event.data.id !== id) return;
      if (settled) return;
      settled = true;
      cleanup();
      resolve(event.data);
    };

    const onError = () => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve({ error: true });
    };

    worker.addEventListener('message', onMessage);
    worker.addEventListener('error', onError);
    worker.postMessage({ ...payload, id, type: 'search' });
  });
}

async function searchMultiThread(board, side, weights, depth, timeLimitMs, movePool = null) {
  const moves = Array.isArray(movePool) ? movePool : generateLegalMoves(board, side);
  if (moves.length === 0) return { move: null, score: -Infinity, depth: 0 };

  const threadCount = Math.max(1, Math.min(state.searchThreads, moves.length));
  if (threadCount <= 1 || depth < MIN_MULTI_DEPTH) {
    return searchBestMoveFromMoves(board, side, weights, depth, {
      moves,
      maxDepth: depth,
      timeLimitMs,
    });
  }

  const groups = Array.from({ length: threadCount }, () => []);
  moves.forEach((move, idx) => {
    groups[idx % threadCount].push(move);
  });

  const tasks = groups.map((group, index) =>
    runWorkerSearch(
      getSearchWorker(index),
      { board, side, weights, depth, maxDepth: depth, timeLimitMs, moves: group },
      timeLimitMs
    )
  );

  const results = await Promise.all(tasks);
  let bestMove = null;
  let bestScore = -Infinity;
  let bestDepth = 0;

  for (const res of results) {
    if (!res || res.error || res.timeout) continue;
    if (res.move && res.score > bestScore) {
      bestScore = res.score;
      bestMove = res.move;
      bestDepth = res.depth || 0;
    }
  }

  if (!bestMove) {
    return searchBestMove(board, side, weights, depth, {
      useBook: false,
      maxDepth: depth,
      timeLimitMs,
    });
  }

  return { move: bestMove, score: bestScore, depth: bestDepth };
}

function renderBoard() {
  const cells = boardEl.querySelectorAll('.cell');
  cells.forEach((cell) => {
    const idx = Number(cell.dataset.idx);
    const piece = state.board[idx];
    cell.textContent = piece ? PIECE_LABELS[piece] || piece : '';
    cell.classList.remove('red', 'black', 'selected', 'move', 'capture', 'last-from', 'last-to', 'checking', 'in-check');
    if (piece) {
      if (isRed(piece)) cell.classList.add('red');
      if (isBlack(piece)) cell.classList.add('black');
    }
  });

  if (state.selected !== null) {
    const selectedCell = boardEl.querySelector(`.cell[data-idx="${state.selected}"]`);
    if (selectedCell) selectedCell.classList.add('selected');
    state.legalMoves.forEach((move) => {
      const targetCell = boardEl.querySelector(`.cell[data-idx="${move.to}"]`);
      if (targetCell) targetCell.classList.add(move.capture ? 'capture' : 'move');
    });
  }

  if (state.lastMove) {
    const fromCell = boardEl.querySelector(`.cell[data-idx="${state.lastMove.from}"]`);
    const toCell = boardEl.querySelector(`.cell[data-idx="${state.lastMove.to}"]`);
    if (fromCell) fromCell.classList.add('last-from');
    if (toCell) toCell.classList.add('last-to');
  }

  const checkers = getCheckers(state.board, state.sideToMove);
  if (checkers.length) {
    checkers.forEach((idx) => {
      const cell = boardEl.querySelector(`.cell[data-idx="${idx}"]`);
      if (cell) cell.classList.add('checking');
    });
    const kingIdx = findKingIndexForSide(state.board, state.sideToMove);
    if (kingIdx !== -1) {
      const kingCell = boardEl.querySelector(`.cell[data-idx="${kingIdx}"]`);
      if (kingCell) kingCell.classList.add('in-check');
    }
  }

  renderTrace();
}

function isHumanTurn() {
  return !state.aiSide[state.sideToMove];
}

function updateLegalMoves() {
  if (state.selected === null) {
    state.legalMoves = [];
    return;
  }
  const moves = generateLegalMoves(state.board, state.sideToMove);
  state.legalMoves = moves.filter((move) => move.from === state.selected);
}

async function onBoardClick(event) {
  const target = event.target;
  const cell = target && target.closest ? target.closest('.cell') : target && target.parentElement ? target.parentElement.closest('.cell') : null;
  if (!cell) return;
  if (state.gameOver) return;
  if (!state.backendOnline) {
    if (state.selected !== null) {
      state.selected = null;
      state.legalMoves = [];
      renderBoard();
    }
    setStatus('后端离线，无法走子');
    return;
  }
  await refreshBackendStatus();
  if (!state.backendModelReady) {
    if (state.selected !== null) {
      state.selected = null;
      state.legalMoves = [];
      renderBoard();
    }
    setStatus('模型未加载，无法走子');
    return;
  }
  if (!isHumanTurn()) return;

  const idx = Number(cell.dataset.idx);
  const piece = state.board[idx];

  if (state.selected !== null) {
    const move = state.legalMoves.find((m) => m.to === idx);
    if (move) {
      commitMove(move, { byAI: false });
      return;
    }
  }

  if (piece && ((state.sideToMove === 'r' && isRed(piece)) || (state.sideToMove === 'b' && isBlack(piece)))) {
    state.selected = idx;
    updateLegalMoves();
    renderBoard();
  } else {
    state.selected = null;
    state.legalMoves = [];
    renderBoard();
  }
}

function initDrawState() {
  const key = positionKey(state.board, state.sideToMove);
  state.positionCounts = new Map([[key, 1]]);
  state.checkStreak = { side: null, count: 0 };
}

function rebuildDrawState() {
  const counts = new Map();
  let checkStreak = { side: null, count: 0 };
  const snapshots = state.history.map((entry) => ({
    board: entry.board,
    sideToMove: entry.sideToMove,
  }));
  snapshots.push({ board: state.board, sideToMove: state.sideToMove });

  let first = true;
  for (const snap of snapshots) {
    const key = positionKey(snap.board, snap.sideToMove);
    counts.set(key, (counts.get(key) || 0) + 1);

    if (!first) {
      const inCheck = getCheckers(snap.board, snap.sideToMove).length > 0;
      if (inCheck) {
        const checkingSide = opposite(snap.sideToMove);
        if (checkStreak.side === checkingSide) checkStreak.count += 1;
        else checkStreak = { side: checkingSide, count: 1 };
      } else {
        checkStreak = { side: null, count: 0 };
      }
    }
    first = false;
  }

  state.positionCounts = counts;
  state.checkStreak = checkStreak;
}

function updateDrawState() {
  const key = positionKey(state.board, state.sideToMove);
  const count = (state.positionCounts.get(key) || 0) + 1;
  state.positionCounts.set(key, count);

  const inCheck = getCheckers(state.board, state.sideToMove).length > 0;
  if (inCheck) {
    const checkingSide = opposite(state.sideToMove);
    if (state.checkStreak.side === checkingSide) state.checkStreak.count += 1;
    else state.checkStreak = { side: checkingSide, count: 1 };
  } else {
    state.checkStreak = { side: null, count: 0 };
  }

  if (count >= REPETITION_LIMIT) return 'repetition';
  if (state.checkStreak.count >= LONG_CHECK_LIMIT) return 'perpetual';
  return null;
}

function wouldRepeatPosition(board, side) {
  const key = positionKey(board, side);
  const count = (state.positionCounts.get(key) || 0) + 1;
  return count >= REPETITION_LIMIT;
}

function filterRepetitionMoves(board, side, moves) {
  if (!moves || moves.length === 0) return [];
  const safe = moves.filter((move) => !wouldRepeatPosition(makeMove(board, move), opposite(side)));
  return safe.length ? safe : moves;
}

function announceGameOver(status) {
  if (status.winner === 'r') {
    if (status.reason === 'no_moves') {
      setStatus('红方胜 (无子可走)');
    } else {
      setStatus('红方胜 (将死)');
    }
    playCheckmateSound();
  } else if (status.winner === 'b') {
    if (status.reason === 'no_moves') {
      setStatus('黑方胜 (无子可走)');
    } else {
      setStatus('黑方胜 (将死)');
    }
    playCheckmateSound();
  } else {
    setStatus('和棋');
  }
}

function commitMove(move, meta = {}) {
  const { byAI = false } = meta;
  state.history.push({
    board: state.board,
    sideToMove: state.sideToMove,
    lastMove: state.lastMove,
    lastMoveByAI: state.lastMoveByAI,
  });
  state.board = makeMove(state.board, move);
  state.sideToMove = opposite(state.sideToMove);
  state.selected = null;
  state.legalMoves = [];
  state.lastMove = move;
  state.lastMoveByAI = byAI;

  const status = gameStatus(state.board, state.sideToMove);
  state.gameOver = status.over;

  if (!status.over) {
    const drawReason = updateDrawState();
    if (drawReason) {
      state.gameOver = true;
      setStatus(drawReason === 'repetition' ? '和棋 (重复局面)' : '和棋 (长将)');
      renderBoard();
      return;
    }
  }

  if (status.over) {
    announceGameOver(status);
  } else {
    setTurnStatus();
  }

  renderBoard();

  if (!state.gameOver && state.autoPlay && state.aiSide[state.sideToMove]) {
    requestAIMove();
  }
}

async function requestPythonMove(board, side, depth, timeLimitMs) {
  const baseUrl = getBackendBaseUrl();
  const sims = pythonSimsForDepth(depth);
  const payload = {
    board,
    side,
    sims,
    temperature: 0.0,
    useBook: state.useOpeningBook,
    timeLimitMs,
  };

  const controller = new AbortController();
  const timeoutMs = Math.max(15000, timeLimitMs * 6 || 0);
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${baseUrl}/ai/move`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!res.ok) {
      const reason = res.status === 409 ? 'model' : 'http';
      return { ok: false, reason };
    }
    const data = await res.json();
    if (!data || typeof data !== 'object' || !('move' in data)) {
      return { ok: false };
    }
    if (!data.move) {
    return { ok: true, move: null };
  }
  return { ok: true, move: { from: data.move.from, to: data.move.to } };
  } catch (err) {
    return { ok: false };
  } finally {
    clearTimeout(timeout);
  }
}

function setBackendStatus(stateLabel, detail, modelDetail = null) {
  if (!backendStatusEl) return;
  const bookTag = state.lastBookMove ? ' · 开局库' : '';
  const parts = [];
  if (detail) parts.push(detail);
  const effectiveModel = modelDetail === null ? state.backendModelLabel : modelDetail;
  if (effectiveModel) parts.push(effectiveModel);
  const detailText = parts.length ? ` · ${parts.join(' · ')}` : '';
  backendStatusEl.textContent = `后端状态：${stateLabel}${detailText}${bookTag}`;
  backendStatusEl.classList.remove('good', 'bad');
  if (stateLabel === '在线') backendStatusEl.classList.add('good');
  if (stateLabel === '离线') backendStatusEl.classList.add('bad');
}

function formatModelLabel(modelInfo) {
  if (!modelInfo) return '';
  if (!modelInfo.loaded) return '模型 未加载';
  const step = Number.isFinite(modelInfo.step) ? `#${modelInfo.step}` : '#?';
  const fileName = modelInfo.path ? String(modelInfo.path).split('/').pop() : 'latest.pt';
  const mtimeMs = Number.isFinite(modelInfo.mtime) ? modelInfo.mtime * 1000 : null;
  if (mtimeMs) {
    return `模型 ${fileName} @ ${formatClockTime(mtimeMs)}`;
  }
  return `模型 ${fileName} ${step}`;
}

async function refreshBackendStatus() {
  if (!backendStatusEl) return;
  const baseUrl = getBackendBaseUrl();
  try {
    const res = await fetch(`${baseUrl}/health?ts=${Date.now()}`, { cache: 'no-store' });
    if (!res.ok) throw new Error('bad');
    const data = await res.json();
    state.backendLastCheck = Date.now();
    const device = data && data.device ? `device ${data.device}` : '';
    const modelInfo = data && data.model ? data.model : null;
    const modelLabel = formatModelLabel(modelInfo);
    if (modelInfo && modelInfo.loaded === false) {
      pythonTrainBaseTotalGames = null;
      state.backendTotalGames = null;
      clearTotalGames();
    }
    const modelTotal = modelInfo && Number.isFinite(modelInfo.total_games) ? modelInfo.total_games : null;
    if (Number.isFinite(modelTotal)) {
      if (!Number.isFinite(pythonTrainBaseTotalGames) || modelTotal >= pythonTrainBaseTotalGames) {
        pythonTrainBaseTotalGames = modelTotal;
        saveTotalGames(modelTotal);
      }
      state.backendTotalGames = modelTotal;
    } else if (Number.isFinite(pythonTrainBaseTotalGames)) {
      state.backendTotalGames = pythonTrainBaseTotalGames;
    }
    state.backendOnline = true;
    state.backendDevice = device;
    state.backendModelLabel = modelLabel;
    state.backendModelReady = Boolean(modelInfo && modelInfo.loaded);
    setBackendStatus('在线', device, modelLabel);
    if (statusEl && statusEl.textContent.includes('后端')) {
      const isFreshStart = !state.gameOver && state.history.length === 0 && !state.lastMove;
      if (isFreshStart) {
        if (state.backendModelReady) {
          setStatus('新对局开始：红方先走');
        } else {
          setStatus('模型未加载，无法走子');
        }
      } else {
        setTurnStatus();
      }
    }
    scheduleTrainCardSync();
    return state.backendModelReady;
  } catch (err) {
    state.backendOnline = false;
    state.backendDevice = '';
    state.backendModelReady = false;
    state.lastBookMove = false;
    state.backendLastCheck = Date.now();
    setBackendStatus('离线', '未连接');
    if (!state.gameOver) {
      setStatus('后端离线，无法走子');
    }
    scheduleTrainCardSync();
    return false;
  }
}

function setPythonTrainControls(running) {
  if (pyTrainStartEl) pyTrainStartEl.disabled = running;
  if (pyTrainStopEl) pyTrainStopEl.disabled = !running;
  if (pyTrainStartEl) {
    if (!pyTrainStartEl.dataset.label) pyTrainStartEl.dataset.label = pyTrainStartEl.textContent || '开始训练';
    if (running) {
      pyTrainStartEl.textContent = pythonTrainStartPending ? '启动中…' : '训练中…';
    } else {
      const original = pyTrainStartEl.dataset.label || '开始训练';
      pyTrainStartEl.textContent = original;
    }
  }
}

const PYTHON_LOG_MAX_LINES = 2;

function parsePythonTrainProgress(lines) {
  if (!Array.isArray(lines)) return null;
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const line = lines[i];
    if (!line || /^progress\s+iter\s+/i.test(line)) continue;
    const match = line.match(/^iter\s*(\d+)\s*\/\s*(\d+)/i);
    if (match) {
      return { current: parseInt(match[1], 10), total: parseInt(match[2], 10) };
    }
  }
  return null;
}

function parsePythonIterProgress(lines) {
  if (!Array.isArray(lines)) return null;
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const match = lines[i].match(/progress\s+iter\s+(\d+)\s*\/\s*(\d+)\s+pct\s+(\d+)/i);
    if (match) {
      return {
        current: parseInt(match[1], 10),
        total: parseInt(match[2], 10),
        pct: parseInt(match[3], 10),
      };
    }
  }
  return null;
}

function updatePythonTrainProgress(lines, config, running, startTime) {
  if (!running) {
    pythonTrainRunning = false;
    pythonTrainStartTime = null;
    pythonTrainProgress = { current: 0, total: 0 };
    pythonTrainIterProgress = { current: 0, total: 0, pct: 0 };
    pythonTrainGamesDone = 0;
    return;
  }

  pythonTrainRunning = true;
  const parsed = parsePythonTrainProgress(lines);
  if (parsed) {
    pythonTrainProgress = parsed;
  } else if (config && typeof config.iterations === 'number') {
    pythonTrainProgress = { ...pythonTrainProgress, total: config.iterations };
  }

  if (config && typeof config.games_per_iter === 'number' && Number.isFinite(pythonTrainProgress.current)) {
    pythonTrainGamesDone = Math.max(0, pythonTrainProgress.current) * config.games_per_iter;
  }

  const iterParsed = parsePythonIterProgress(lines);
  if (iterParsed) {
    pythonTrainIterProgress = iterParsed;
  } else if (config && typeof config.iterations === 'number' && pythonTrainIterProgress.total === 0) {
    pythonTrainIterProgress = { current: 1, total: config.iterations, pct: 0 };
  }

  if (Number.isFinite(startTime) && startTime > 0) {
    pythonTrainStartTime = startTime * 1000;
  }

  if (!pythonTrainStartTime) {
    pythonTrainStartTime = Date.now();
  }
}

function formatClockTime(ts) {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return '';
  const yyyy = String(d.getFullYear());
  const mmDate = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  return `${yyyy}-${mmDate}-${dd} ${hh}:${mm}:${ss}`;
}

function getPythonTrainEtaText() {
  if (!pythonTrainRunning) return '';
  const startText = pythonTrainStartTime ? `开始：${formatClockTime(pythonTrainStartTime)} ` : '';
  if (!pythonTrainStartTime || pythonTrainProgress.current <= 0 || pythonTrainProgress.total <= 0) {
    return `${startText}预计剩余：估算中`.trim();
  }
  const elapsed = Math.max(0.001, (Date.now() - pythonTrainStartTime) / 1000);
  if (elapsed < 2) return `${startText}预计剩余：估算中`.trim();
  const rate = pythonTrainProgress.current / elapsed;
  if (!Number.isFinite(rate) || rate <= 0) return `${startText}预计剩余：估算中`.trim();
  const remaining = Math.max(0, pythonTrainProgress.total - pythonTrainProgress.current);
  return `${startText}预计剩余：${formatDuration(remaining / rate)}`.trim();
}

function renderPythonTrainLog() {
  if (!pyTrainLogEl) return;
  pyTrainLogEl.textContent = pythonTrainLogLines.join('\n');

  if (pyTrainLogMetaEl) {
    const eta = getPythonTrainEtaText();
    const lines = [];
    if (eta) lines.push(eta);
    const stats = [];
    if (pythonTrainRunning) stats.push(`本次对局：${pythonTrainGamesDone}`);
    const baseTotal = Number.isFinite(state.backendTotalGames) ? state.backendTotalGames : pythonTrainBaseTotalGames;
    if (Number.isFinite(baseTotal)) {
      const total = pythonTrainRunning ? baseTotal + pythonTrainGamesDone : baseTotal;
      stats.push(`累计对局：${total}`);
    }
    if (stats.length) lines.push(stats.join(' · '));

    pyTrainLogMetaEl.innerHTML = '';
    pyTrainLogMetaEl.classList.toggle('is-two-lines', lines.length >= 2);
    if (lines.length === 1) {
      const div = document.createElement('div');
      div.textContent = lines[0];
      pyTrainLogMetaEl.appendChild(div);
    } else if (lines.length >= 2) {
      for (const line of lines) {
        const div = document.createElement('div');
        div.textContent = line;
        pyTrainLogMetaEl.appendChild(div);
      }
    }
  }

  if (pyTrainIterProgressEl) {
    const fill = pyTrainIterProgressEl.querySelector('.progress-fill');
    const iterEl = pyTrainIterProgressEl.querySelector('.progress-iter');
    const pctEl = pyTrainIterProgressEl.querySelector('.progress-pct');
    const pct = Math.max(0, Math.min(100, pythonTrainIterProgress.pct || 0));
    if (fill) fill.style.width = `${pct}%`;
    if (iterEl) {
      const current = Math.max(0, pythonTrainIterProgress.current || 0);
      const total = Math.max(0, pythonTrainIterProgress.total || 0);
      iterEl.textContent = `迭代 ${current}/${total}`;
    }
    if (pctEl) pctEl.textContent = `${pct.toFixed(1)}%`;
  }

  scheduleTrainCardSync();
}

function scheduleTrainCardSync() {
  if (pythonTrainSyncPending) return;
  pythonTrainSyncPending = true;
  const runner = () => {
    pythonTrainSyncPending = false;
    syncTrainCardBottom();
  };
  if (typeof requestAnimationFrame === 'function') {
    requestAnimationFrame(runner);
  } else {
    setTimeout(runner, 0);
  }
}

function syncTrainCardBottom() {
  if (!statusEl || !pyTrainLogEl) return;
  const trainCard = document.getElementById('trainCard');
  if (!trainCard) return;
  if (trainCard.style.transform) {
    trainCard.style.transform = '';
  }
  const leftBottom = statusEl.getBoundingClientRect().bottom;
  const rightBottom = trainCard.getBoundingClientRect().bottom;
  const delta = leftBottom - rightBottom;
  if (Math.abs(delta) < 0.25) {
    pythonTrainSyncAttempts = 0;
    return;
  }
  const current = pyTrainLogEl.getBoundingClientRect().height || 0;
  const target = Math.max(80, current + delta);
  pyTrainLogEl.style.height = `${target.toFixed(2)}px`;
  if (pythonTrainSyncAttempts < 8) {
    pythonTrainSyncAttempts += 1;
    scheduleTrainCardSync();
  } else {
    pythonTrainSyncAttempts = 0;
  }
}

function setPythonTrainLog(value) {
  if (Array.isArray(value)) {
    pythonTrainLogLines = value.filter((line) => !(line && /^progress\s+iter\s+/i.test(line)));
  } else if (value) {
    pythonTrainLogLines = String(value)
      .split('\n')
      .filter((line) => !(line && /^progress\s+iter\s+/i.test(line)));
  } else {
    pythonTrainLogLines = [];
  }
  renderPythonTrainLog();
}

async function fetchPythonTrainStatus() {
  if (!pyTrainLogEl) return null;
  const baseUrl = getBackendBaseUrl();
  try {
    const timeoutMs = 4000;
    const hasAbort = typeof AbortController !== 'undefined';
    const controller = hasAbort ? new AbortController() : null;
    const fetchPromise = fetch(`${baseUrl}/train/status`, {
      signal: controller ? controller.signal : undefined,
    });
    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(() => {
        if (controller) controller.abort();
        reject(new Error('timeout'));
      }, timeoutMs);
    });
    const res = await Promise.race([fetchPromise, timeoutPromise]);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    const msg =
      err && (err.name === 'AbortError' || err.message === 'timeout')
        ? '训练状态请求超时，请检查后端连接。'
        : '无法连接训练后端。';
    setPythonTrainLog(msg);
    return null;
  }
}

function startPythonTrainPolling() {
  if (pythonTrainTimer) return;
  pythonTrainTimer = setInterval(async () => {
    const status = await fetchPythonTrainStatus();
    if (!status) {
      if (pythonTrainStartPending && pythonTrainStartRequestedAt && Date.now() - pythonTrainStartRequestedAt > 10000) {
        pythonTrainStartPending = false;
        pythonTrainRunning = false;
        setPythonTrainControls(false);
        setPythonTrainLog('启动训练失败：后端未收到启动请求。');
        stopPythonTrainPolling();
      }
      return;
    }
    if (status.running) {
      pythonTrainPostStopChecks = 0;
    }
    if (pythonTrainStartPending) {
      if (status.running) {
        pythonTrainStartPending = false;
      } else if (pythonTrainStartRequestedAt && Date.now() - pythonTrainStartRequestedAt > 10000) {
        pythonTrainStartPending = false;
        pythonTrainRunning = false;
        setPythonTrainControls(false);
        setPythonTrainLog('启动训练失败：后端未收到启动请求。');
        stopPythonTrainPolling();
        return;
      }
    }
    updatePythonTrainProgress(status.lines, status.config, Boolean(status.running), status.startTime);
    if (Array.isArray(status.lines) && status.lines.length) {
      setPythonTrainLog(status.lines);
    } else if (status.running) {
      setPythonTrainLog('训练中（首轮耗时较长，请耐心等待…）');
    }
    setPythonTrainControls(Boolean(status.running));
    if (!status.running && !pythonTrainStartPending) {
      if (pythonTrainPostStopChecks === 0) pythonTrainPostStopChecks = 3;
      pythonTrainPostStopChecks -= 1;
      refreshBackendStatus();
      if (pythonTrainPostStopChecks <= 0) {
        stopPythonTrainPolling();
      }
    }
  }, 2000);
}

function stopPythonTrainPolling() {
  if (pythonTrainTimer) {
    clearInterval(pythonTrainTimer);
    pythonTrainTimer = null;
  }
}

function readPythonTrainInputs() {
  const iterations = clampInt(pyTrainItersEl?.value, 1, 5000, pythonTrainConfig.iterations);
  const gamesPerIter = clampInt(pyTrainGamesEl?.value, 1, 200, pythonTrainConfig.gamesPerIter);
  const simulations = clampInt(pyTrainSimsEl?.value, 20, 4000, pythonTrainConfig.simulations);
  const lr = clampFloat(pyTrainLrEl?.value, 0.0005, 0.01, pythonTrainConfig.lr);
  const batchSize = clampInt(pyTrainBatchEl?.value, 16, 512, pythonTrainConfig.batchSize);
  const epochs = clampInt(pyTrainEpochsEl?.value, 1, 6, pythonTrainConfig.epochs);
  const bufferSize = clampInt(pyTrainBufferEl?.value, 2000, 50000, pythonTrainConfig.bufferSize);

  if (pyTrainItersEl) pyTrainItersEl.value = iterations;
  if (pyTrainGamesEl) pyTrainGamesEl.value = gamesPerIter;
  if (pyTrainSimsEl) pyTrainSimsEl.value = simulations;
  if (pyTrainLrEl) pyTrainLrEl.value = lr;
  if (pyTrainBatchEl) pyTrainBatchEl.value = batchSize;
  if (pyTrainEpochsEl) pyTrainEpochsEl.value = epochs;
  if (pyTrainBufferEl) pyTrainBufferEl.value = bufferSize;

  savePythonTrainConfig({ iterations, gamesPerIter, simulations, lr, batchSize, epochs, bufferSize });
  return { iterations, gamesPerIter, simulations, lr, batchSize, epochs, bufferSize };
}

async function startPythonTraining() {
  if (!pyTrainStartEl) return;
  const { iterations, gamesPerIter, simulations, lr, batchSize, epochs, bufferSize } = readPythonTrainInputs();
  pythonTrainStartTime = Date.now();
  pythonTrainProgress = { current: 0, total: iterations };
  pythonTrainIterProgress = { current: 1, total: iterations, pct: 0 };
  pythonTrainRunning = true;
  pythonTrainStartPending = true;
  pythonTrainStartRequestedAt = Date.now();
  if (Number.isFinite(state.backendTotalGames)) {
    pythonTrainBaseTotalGames = state.backendTotalGames;
  }
  pythonTrainGamesDone = 0;
  setPythonTrainControls(true);
  if (pyTrainStartEl) {
    if (!pyTrainStartEl.dataset.label) pyTrainStartEl.dataset.label = pyTrainStartEl.textContent || '开始训练';
    pyTrainStartEl.textContent = '启动中…';
  }
  setPythonTrainLog('已发送启动请求，等待后端响应…');
  startPythonTrainPolling();
  const baseUrl = getBackendBaseUrl();
  const hasAbort = typeof AbortController !== 'undefined';
  const controller = hasAbort ? new AbortController() : null;
  const timeoutMs = 10000;
  const timeout = setTimeout(() => {
    if (controller) controller.abort();
  }, timeoutMs);
  try {
    const res = await fetch(`${baseUrl}/train/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        iterations,
        games_per_iter: gamesPerIter,
        simulations,
        lr,
        batch_size: batchSize,
        epochs,
        buffer_size: bufferSize,
      }),
      signal: controller ? controller.signal : undefined,
    });
    if (!res.ok) {
      let detail = '';
      try {
        const data = await res.json();
        if (data && typeof data.detail === 'string') detail = data.detail;
      } catch (_err) {
        // ignore parse errors
      }
      const msg = detail ? `HTTP ${res.status}: ${detail}` : `HTTP ${res.status}`;
      throw new Error(msg);
    }
  } catch (err) {
    if (err && err.name === 'AbortError') {
      setPythonTrainLog('启动训练超时，请确认后端可访问。');
    } else {
      const msg = err && err.message ? `启动训练失败：${err.message}` : '启动训练失败，请确认后端已启动。';
      setPythonTrainLog(msg);
    }
    pythonTrainRunning = false;
    pythonTrainStartPending = false;
    setPythonTrainControls(false);
  } finally {
    clearTimeout(timeout);
    setTimeout(async () => {
      const status = await fetchPythonTrainStatus();
      if (!status) return;
      const running = Boolean(status.running);
      if (!running) {
        pythonTrainRunning = false;
        pythonTrainStartPending = false;
        setPythonTrainControls(false);
        if (pythonTrainLogLines.length === 0) {
          setPythonTrainLog('启动训练失败：后端未收到启动请求。');
        }
      }
    }, 12000);
  }
}

async function stopPythonTraining() {
  if (!pyTrainStopEl) return;
  const baseUrl = getBackendBaseUrl();
  try {
    await fetch(`${baseUrl}/train/stop`, { method: 'POST' });
    setPythonTrainLog('已请求停止训练，等待进程退出…');
    setTimeout(refreshBackendStatus, 1200);
    setTimeout(refreshBackendStatus, 3200);
  } catch (err) {
    const msg = err && err.message ? `停止训练失败：${err.message}` : '停止训练失败，请确认后端可访问。';
    setPythonTrainLog(msg);
  }
}

async function initPythonTrainingUI() {
  if (!pyTrainLogEl) return;
  const status = await fetchPythonTrainStatus();
  if (!status) return;
  updatePythonTrainProgress(status.lines, status.config, Boolean(status.running), status.startTime);
  if (Array.isArray(status.lines) && status.lines.length) {
    setPythonTrainLog(status.lines);
  } else if (status.running) {
    setPythonTrainLog('训练中（首轮耗时较长，请耐心等待…）');
  }
  setPythonTrainControls(Boolean(status.running));
  if (status.running) {
    startPythonTrainPolling();
  }
  scheduleTrainCardSync();
}

async function requestAIMove() {
  if (state.aiThinking || state.gameOver) return;
  await refreshBackendStatus();
  if (!state.backendModelReady) {
    setStatus('模型未加载，无法走子');
    return;
  }
  state.aiThinking = true;

  const depth = parseInt(depthEl.value, 10);
  const timeLimitMs = AI_TIME_LIMITS[depth] || 700;
  setStatus('Python AI 思考中…');

  const boardSnapshot = state.board;
  const sideSnapshot = state.sideToMove;

  setTimeout(async () => {
    if (state.board !== boardSnapshot || state.sideToMove !== sideSnapshot || state.gameOver) {
      state.aiThinking = false;
      return;
    }

    const pythonResult = await requestPythonMove(state.board, state.sideToMove, depth, timeLimitMs);
    if (!pythonResult || !pythonResult.ok) {
      state.aiThinking = false;
      if (pythonResult && pythonResult.reason === 'model') {
        state.backendModelReady = false;
        state.lastBookMove = false;
        setBackendStatus('在线', state.backendDevice || '', state.backendModelLabel);
        setStatus('模型未加载，无法走子');
        return;
      }
      state.backendOnline = false;
      state.backendDevice = '';
      state.lastBookMove = false;
      setBackendStatus('离线', '走子失败');
      setStatus('后端不可用');
      return;
    }
    state.lastBookMove = Boolean(pythonResult.book);
    if (state.backendOnline) {
      setBackendStatus('在线', state.backendDevice || '');
    }
    if (!pythonResult.move) {
      state.aiThinking = false;
      const status = gameStatus(state.board, state.sideToMove);
      if (status.over) {
        state.gameOver = true;
        announceGameOver(status);
        renderBoard();
      } else {
        setStatus('后端未返回着法');
      }
      return;
    }
    state.aiThinking = false;
    commitMove(pythonResult.move, { byAI: true });
  }, 30);
}

function resetGame() {
  state.board = initialBoard();
  state.sideToMove = 'r';
  state.selected = null;
  state.legalMoves = [];
  state.history = [];
  state.lastMove = null;
  state.lastMoveByAI = false;
  state.gameOver = false;
  state.useOpeningBook = openingBookEl ? openingBookEl.checked : true;
  initDrawState();
  if (state.backendOnline) {
    if (state.backendModelReady) {
      setStatus('新对局开始：红方先走');
    } else {
      setStatus('模型未加载，无法走子');
    }
  } else {
    setStatus('后端检测中…');
  }
  renderBoard();
  if (state.autoPlay && state.aiSide[state.sideToMove]) {
    requestAIMove();
  }
}

function undoMove() {
  const last = state.history.pop();
  if (!last) return;
  state.board = last.board;
  state.sideToMove = last.sideToMove;
  state.lastMove = last.lastMove;
  state.lastMoveByAI = last.lastMoveByAI || false;
  state.selected = null;
  state.legalMoves = [];
  state.gameOver = false;
  rebuildDrawState();
  setTurnStatus();
  renderBoard();
}

function bindControls() {
  aiRedEl.addEventListener('change', () => {
    state.aiSide.r = aiRedEl.checked;
    if (state.autoPlay && state.aiSide[state.sideToMove]) requestAIMove();
  });

  aiBlackEl.addEventListener('change', () => {
    state.aiSide.b = aiBlackEl.checked;
    if (state.autoPlay && state.aiSide[state.sideToMove]) requestAIMove();
  });

  autoPlayEl.addEventListener('change', () => {
    state.autoPlay = autoPlayEl.checked;
    if (state.autoPlay && state.aiSide[state.sideToMove]) requestAIMove();
  });

  if (openingBookEl) {
    state.useOpeningBook = openingBookEl.checked;
    openingBookEl.addEventListener('change', () => {
      state.useOpeningBook = openingBookEl.checked;
    });
  }

  if (pyTrainItersEl) {
    pyTrainItersEl.value = pythonTrainConfig.iterations;
    pyTrainItersEl.addEventListener('change', () => {
      const iterations = clampInt(pyTrainItersEl.value, 1, 5000, pythonTrainConfig.iterations);
      pyTrainItersEl.value = iterations;
      savePythonTrainConfig({ iterations });
    });
  }

  if (pyTrainGamesEl) {
    pyTrainGamesEl.value = pythonTrainConfig.gamesPerIter;
    pyTrainGamesEl.addEventListener('change', () => {
      const gamesPerIter = clampInt(pyTrainGamesEl.value, 1, 200, pythonTrainConfig.gamesPerIter);
      pyTrainGamesEl.value = gamesPerIter;
      savePythonTrainConfig({ gamesPerIter });
    });
  }

  if (pyTrainSimsEl) {
    pyTrainSimsEl.value = pythonTrainConfig.simulations;
    pyTrainSimsEl.addEventListener('change', () => {
      const simulations = clampInt(pyTrainSimsEl.value, 20, 4000, pythonTrainConfig.simulations);
      pyTrainSimsEl.value = simulations;
      savePythonTrainConfig({ simulations });
    });
  }

  if (pyTrainAdvancedEl && pyTrainAdvancedPanelEl) {
    pyTrainAdvancedEl.checked = Boolean(pythonTrainConfig.advanced);
    pyTrainAdvancedPanelEl.hidden = !pyTrainAdvancedEl.checked;
    pyTrainAdvancedEl.addEventListener('change', () => {
      const enabled = pyTrainAdvancedEl.checked;
      pyTrainAdvancedPanelEl.hidden = !enabled;
      savePythonTrainConfig({ advanced: enabled });
    });
  }

  if (pyTrainPresetEl) {
    const presetValue = pythonTrainConfig.preset || 'balanced';
    pyTrainPresetEl.value = presetValue;
    pyTrainPresetEl.addEventListener('change', () => {
      const key = pyTrainPresetEl.value;
      if (key === 'custom') {
        savePythonTrainConfig({ preset: 'custom' });
        return;
      }
      applyPythonTrainPreset(key);
    });
  }

  if (pyTrainLrEl) {
    pyTrainLrEl.value = pythonTrainConfig.lr;
    pyTrainLrEl.addEventListener('change', () => {
      const lr = clampFloat(pyTrainLrEl.value, 0.0005, 0.01, pythonTrainConfig.lr);
      pyTrainLrEl.value = lr;
      savePythonTrainConfig({ lr });
      setPresetCustom();
    });
  }

  if (pyTrainBatchEl) {
    pyTrainBatchEl.value = pythonTrainConfig.batchSize;
    pyTrainBatchEl.addEventListener('change', () => {
      const batchSize = clampInt(pyTrainBatchEl.value, 16, 512, pythonTrainConfig.batchSize);
      pyTrainBatchEl.value = batchSize;
      savePythonTrainConfig({ batchSize });
      setPresetCustom();
    });
  }

  if (pyTrainEpochsEl) {
    pyTrainEpochsEl.value = pythonTrainConfig.epochs;
    pyTrainEpochsEl.addEventListener('change', () => {
      const epochs = clampInt(pyTrainEpochsEl.value, 1, 6, pythonTrainConfig.epochs);
      pyTrainEpochsEl.value = epochs;
      savePythonTrainConfig({ epochs });
      setPresetCustom();
    });
  }

  if (pyTrainBufferEl) {
    pyTrainBufferEl.value = pythonTrainConfig.bufferSize;
    pyTrainBufferEl.addEventListener('change', () => {
      const bufferSize = clampInt(pyTrainBufferEl.value, 2000, 50000, pythonTrainConfig.bufferSize);
      pyTrainBufferEl.value = bufferSize;
      savePythonTrainConfig({ bufferSize });
      setPresetCustom();
    });
  }

  if (pyTrainStartEl) {
    pyTrainStartEl.addEventListener('click', startPythonTraining);
  }

  if (pyTrainStopEl) {
    pyTrainStopEl.addEventListener('click', stopPythonTraining);
  }

  if (pyTrainLogToggleEl) {
    pyTrainLogToggleEl.addEventListener('click', () => {
      pythonTrainLogExpanded = !pythonTrainLogExpanded;
      renderPythonTrainLog();
    });
  }

  newGameEl.addEventListener('click', resetGame);
  undoEl.addEventListener('click', undoMove);
  aiMoveEl.addEventListener('click', () => {
    if (!state.aiSide[state.sideToMove]) {
      state.aiSide[state.sideToMove] = true;
      if (state.sideToMove === 'r') aiRedEl.checked = true;
      else aiBlackEl.checked = true;
    }
    requestAIMove();
  });

  window.addEventListener('resize', renderTrace);
  window.addEventListener('resize', scheduleTrainCardSync);
  window.addEventListener('load', scheduleTrainCardSync);
}

createBoard();
bindControls();
refreshBackendStatus();
setInterval(refreshBackendStatus, 1000);
initPythonTrainingUI();
renderPythonTrainLog();
resetGame();
