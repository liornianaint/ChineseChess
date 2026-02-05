let shouldStop = false;

import {
  initialBoard,
  generateLegalMoves,
  makeMove,
  searchBestMove,
  updateWeightsFromHistory,
  cloneWeights,
  defaultWeights,
  opposite,
  gameStatus,
  isInCheck,
  positionKey,
} from './engine.js';

const REPETITION_LIMIT = 3;
const LONG_CHECK_LIMIT = 6;
const YIELD_EVERY = 4;

function yieldToEventLoop() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function pickMove(board, side, weights, depth, epsilon, timeLimitMs) {
  const moves = generateLegalMoves(board, side);
  if (moves.length === 0) return null;
  if (Math.random() < epsilon) {
    return moves[Math.floor(Math.random() * moves.length)];
  }
  const { move } = searchBestMove(board, side, weights, depth, { timeLimitMs, maxDepth: depth });
  return move || moves[0];
}

function createDrawState(board, side) {
  const key = positionKey(board, side);
  return {
    counts: new Map([[key, 1]]),
    checkStreak: { side: null, count: 0 },
  };
}

function updateDrawState(drawState, board, side) {
  const key = positionKey(board, side);
  const count = (drawState.counts.get(key) || 0) + 1;
  drawState.counts.set(key, count);

  const inCheck = isInCheck(board, side);
  if (inCheck) {
    const checkingSide = opposite(side);
    if (drawState.checkStreak.side === checkingSide) drawState.checkStreak.count += 1;
    else drawState.checkStreak = { side: checkingSide, count: 1 };
  } else {
    drawState.checkStreak = { side: null, count: 0 };
  }

  if (count >= REPETITION_LIMIT) return 'repetition';
  if (drawState.checkStreak.count >= LONG_CHECK_LIMIT) return 'perpetual';
  return null;
}

async function selfPlayGame(weights, options) {
  const history = [];
  let board = initialBoard();
  let side = 'r';
  const maxPlies = options.maxPlies || 200;
  const depth = options.depth || 2;
  const timeLimitMs = options.timeLimitMs || 0;
  const epsilon = options.epsilon || 0.12;
  const drawState = createDrawState(board, side);

  for (let ply = 0; ply < maxPlies; ply += 1) {
    if (shouldStop) return { result: 0, history, plies: ply, stopped: true };
    if (ply % YIELD_EVERY === 0) {
      await yieldToEventLoop();
      if (shouldStop) return { result: 0, history, plies: ply, stopped: true };
    }
    history.push(board);
    const move = pickMove(board, side, weights, depth, epsilon, timeLimitMs);
    if (!move) break;
    board = makeMove(board, move);
    side = opposite(side);
    const status = gameStatus(board, side);
    if (status.over) {
      const result = status.winner === null ? 0 : status.winner === 'r' ? 1 : -1;
      return { result, history, plies: ply + 1, stopped: false };
    }

    const drawReason = updateDrawState(drawState, board, side);
    if (drawReason) {
      return { result: 0, history, plies: ply + 1, stopped: false };
    }
  }

  return { result: 0, history, plies: maxPlies, stopped: false };
}

self.onmessage = async (event) => {
  const { type } = event.data;
  if (type === 'stop') {
    shouldStop = true;
    return;
  }
  if (type !== 'train') return;
  shouldStop = false;

  const games = event.data.games || 20;
  const depth = event.data.depth || 2;
  const lr = event.data.lr || 0.02;
  const maxPlies = event.data.maxPlies || 200;
  const timeLimitMs = event.data.timeLimitMs || 0;

  const weights = event.data.weights ? cloneWeights(event.data.weights) : defaultWeights();
  const stats = { games: 0, red: 0, black: 0, draw: 0, plies: 0 };

  for (let i = 0; i < games; i += 1) {
    if (shouldStop) break;
    const outcome = await selfPlayGame(weights, {
      depth,
      maxPlies,
      epsilon: 0.12,
      timeLimitMs,
    });

    if (outcome.stopped) {
      shouldStop = true;
      break;
    }

    const { result, history, plies } = outcome;
    updateWeightsFromHistory(weights, history, result, lr);
    stats.games += 1;
    stats.plies += plies;
    if (result === 1) stats.red += 1;
    else if (result === -1) stats.black += 1;
    else stats.draw += 1;

    if ((i + 1) % 1 === 0 || i === games - 1) {
      self.postMessage({
        type: 'progress',
        stats: { ...stats },
        current: i + 1,
        total: games,
      });
    }

    if (i % YIELD_EVERY === 0) {
      await yieldToEventLoop();
    }
  }

  self.postMessage({
    type: 'done',
    stats,
    weights,
    stopped: shouldStop,
  });
};
