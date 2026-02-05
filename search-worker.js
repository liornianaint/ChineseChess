import { searchBestMoveFromMoves } from './engine.js';

self.onmessage = (event) => {
  const { type } = event.data || {};
  if (type !== 'search') return;

  const {
    id,
    board,
    side,
    weights,
    depth,
    moves,
    timeLimitMs,
    maxDepth,
    ttMax,
  } = event.data;

  const result = searchBestMoveFromMoves(board, side, weights, depth, {
    moves,
    timeLimitMs,
    maxDepth,
    ttMax,
  });

  self.postMessage({ id, ...result });
};
