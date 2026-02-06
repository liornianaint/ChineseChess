const FILES = 9;
const RANKS = 10;
const BOARD_SIZE = FILES * RANKS;
const MATE_SCORE = 100000;
const EVAL = {
  MOBILITY_WEIGHT: 2,
  CHECK_PENALTY: 90,
  CONNECTED_PAWN: 12,
  ISOLATED_PAWN: 8,
  PASSED_PAWN: 14,
  PAWN_CROSSED: 10,
  PAWN_ADVANCE: 4,
  OPEN_FILE: 12,
  OPEN_FILE_FULL: 18,
  PALACE_PRESSURE: 6,
};

const PIECE_TYPES = ['K', 'A', 'B', 'N', 'R', 'C', 'P'];

const HORSE_DIRS = [
  { dx: 2, dy: 1, lx: 1, ly: 0 },
  { dx: 2, dy: -1, lx: 1, ly: 0 },
  { dx: -2, dy: 1, lx: -1, ly: 0 },
  { dx: -2, dy: -1, lx: -1, ly: 0 },
  { dx: 1, dy: 2, lx: 0, ly: 1 },
  { dx: -1, dy: 2, lx: 0, ly: 1 },
  { dx: 1, dy: -2, lx: 0, ly: -1 },
  { dx: -1, dy: -2, lx: 0, ly: -1 },
];

const ELE_DIRS = [
  { dx: 2, dy: 2, ex: 1, ey: 1 },
  { dx: 2, dy: -2, ex: 1, ey: -1 },
  { dx: -2, dy: 2, ex: -1, ey: 1 },
  { dx: -2, dy: -2, ex: -1, ey: -1 },
];

const ADVISOR_DIRS = [
  { dx: 1, dy: 1 },
  { dx: 1, dy: -1 },
  { dx: -1, dy: 1 },
  { dx: -1, dy: -1 },
];

const KING_DIRS = [
  { dx: 1, dy: 0 },
  { dx: -1, dy: 0 },
  { dx: 0, dy: 1 },
  { dx: 0, dy: -1 },
];

const ROOK_DIRS = [
  { dx: 1, dy: 0 },
  { dx: -1, dy: 0 },
  { dx: 0, dy: 1 },
  { dx: 0, dy: -1 },
];

function coordToIndex(x, y) {
  return y * FILES + x;
}

function indexToCoord(idx) {
  return { x: idx % FILES, y: Math.floor(idx / FILES) };
}

function inBounds(x, y) {
  return x >= 0 && x < FILES && y >= 0 && y < RANKS;
}

function isRed(piece) {
  return piece && piece === piece.toUpperCase();
}

function isBlack(piece) {
  return piece && piece === piece.toLowerCase();
}

function pieceType(piece) {
  return piece.toUpperCase();
}

function sideOf(piece) {
  return isRed(piece) ? 'r' : 'b';
}

function opposite(side) {
  return side === 'r' ? 'b' : 'r';
}

function mirrorIndex(idx) {
  const { x, y } = indexToCoord(idx);
  return coordToIndex(x, RANKS - 1 - y);
}

function positionKey(board, side) {
  return `${side}:${board.map((piece) => piece || '.').join('')}`;
}

function moveKey(move) {
  return `${move.from}-${move.to}`;
}

function nowMs() {
  if (typeof performance !== 'undefined' && performance.now) return performance.now();
  return Date.now();
}

function moveOrderScore(move, weights) {
  let score = 0;
  if (move.capture) {
    const captureType = pieceType(move.capture);
    score += 10000 + (weights?.values?.[captureType] || 0);
  }
  const moverType = pieceType(move.piece);
  score += (weights?.values?.[moverType] || 0) * 0.02;
  return score;
}

function orderMoves(moves, ttMove, weights) {
  if (moves.length < 2) return moves;
  const ttKey = ttMove ? moveKey(ttMove) : null;
  moves.sort((a, b) => {
    const aIsTt = ttKey && moveKey(a) === ttKey ? 1 : 0;
    const bIsTt = ttKey && moveKey(b) === ttKey ? 1 : 0;
    if (aIsTt !== bIsTt) return bIsTt - aIsTt;
    return moveOrderScore(b, weights) - moveOrderScore(a, weights);
  });
  return moves;
}

function inPalace(x, y, side) {
  if (x < 3 || x > 5) return false;
  if (side === 'r') return y >= 7 && y <= 9;
  return y >= 0 && y <= 2;
}

function initialBoard() {
  const board = Array(BOARD_SIZE).fill(null);
  const placeRow = (y, row) => {
    for (let x = 0; x < FILES; x += 1) {
      const piece = row[x];
      if (piece) {
        board[coordToIndex(x, y)] = piece;
      }
    }
  };

  placeRow(0, ['r', 'n', 'b', 'a', 'k', 'a', 'b', 'n', 'r']);
  placeRow(1, [null, null, null, null, null, null, null, null, null]);
  placeRow(2, [null, 'c', null, null, null, null, null, 'c', null]);
  placeRow(3, ['p', null, 'p', null, 'p', null, 'p', null, 'p']);
  placeRow(4, [null, null, null, null, null, null, null, null, null]);
  placeRow(5, [null, null, null, null, null, null, null, null, null]);
  placeRow(6, ['P', null, 'P', null, 'P', null, 'P', null, 'P']);
  placeRow(7, [null, 'C', null, null, null, null, null, 'C', null]);
  placeRow(8, [null, null, null, null, null, null, null, null, null]);
  placeRow(9, ['R', 'N', 'B', 'A', 'K', 'A', 'B', 'N', 'R']);

  return board;
}

function defaultWeights() {
  const values = {
    K: 10000,
    A: 220,
    B: 240,
    N: 320,
    R: 520,
    C: 480,
    P: 110,
  };

  const pst = {};
  for (const type of PIECE_TYPES) {
    pst[type] = Array(BOARD_SIZE).fill(0);
  }

  // Small forward incentive for pawns
  for (let y = 0; y < RANKS; y += 1) {
    for (let x = 0; x < FILES; x += 1) {
      const idx = coordToIndex(x, y);
      const advance = RANKS - 1 - y;
      pst.P[idx] = Math.max(0, 5 - Math.abs(x - 4)) + advance * 2;
    }
  }

  return { values, pst, version: 1 };
}

function cloneWeights(weights) {
  const next = { values: {}, pst: {}, version: weights.version || 1 };
  for (const type of PIECE_TYPES) {
    next.values[type] = weights.values[type];
    next.pst[type] = weights.pst[type].slice();
  }
  return next;
}

function clampWeights(weights) {
  for (const type of PIECE_TYPES) {
    const val = weights.values[type];
    weights.values[type] = Math.max(-20000, Math.min(20000, val));
    const table = weights.pst[type];
    for (let i = 0; i < table.length; i += 1) {
      table[i] = Math.max(-500, Math.min(500, table[i]));
    }
  }
}

function addMove(moves, board, from, to) {
  const target = board[to];
  if (target && sideOf(target) === sideOf(board[from])) return;
  moves.push({
    from,
    to,
    piece: board[from],
    capture: target || null,
  });
}

function generatePseudoMoves(board, side) {
  const moves = [];

  for (let idx = 0; idx < BOARD_SIZE; idx += 1) {
    const piece = board[idx];
    if (!piece) continue;
    if ((side === 'r' && !isRed(piece)) || (side === 'b' && !isBlack(piece))) {
      continue;
    }

    const { x, y } = indexToCoord(idx);
    const type = pieceType(piece);

    if (type === 'R') {
      for (const dir of ROOK_DIRS) {
        let nx = x + dir.dx;
        let ny = y + dir.dy;
        while (inBounds(nx, ny)) {
          const to = coordToIndex(nx, ny);
          if (!board[to]) {
            addMove(moves, board, idx, to);
          } else {
            if (sideOf(board[to]) !== side) {
              addMove(moves, board, idx, to);
            }
            break;
          }
          nx += dir.dx;
          ny += dir.dy;
        }
      }
    } else if (type === 'C') {
      for (const dir of ROOK_DIRS) {
        let nx = x + dir.dx;
        let ny = y + dir.dy;
        let screen = false;
        while (inBounds(nx, ny)) {
          const to = coordToIndex(nx, ny);
          const target = board[to];
          if (!screen) {
            if (!target) {
              addMove(moves, board, idx, to);
            } else {
              screen = true;
            }
          } else if (target) {
            if (sideOf(target) !== side) {
              addMove(moves, board, idx, to);
            }
            break;
          }
          nx += dir.dx;
          ny += dir.dy;
        }
      }
    } else if (type === 'N') {
      for (const dir of HORSE_DIRS) {
        const lx = x + dir.lx;
        const ly = y + dir.ly;
        if (!inBounds(lx, ly)) continue;
        if (board[coordToIndex(lx, ly)]) continue;
        const nx = x + dir.dx;
        const ny = y + dir.dy;
        if (!inBounds(nx, ny)) continue;
        const to = coordToIndex(nx, ny);
        if (!board[to] || sideOf(board[to]) !== side) {
          addMove(moves, board, idx, to);
        }
      }
    } else if (type === 'B') {
      for (const dir of ELE_DIRS) {
        const nx = x + dir.dx;
        const ny = y + dir.dy;
        const ex = x + dir.ex;
        const ey = y + dir.ey;
        if (!inBounds(nx, ny) || !inBounds(ex, ey)) continue;
        if (board[coordToIndex(ex, ey)]) continue;
        if (side === 'r' && ny < 5) continue;
        if (side === 'b' && ny > 4) continue;
        const to = coordToIndex(nx, ny);
        if (!board[to] || sideOf(board[to]) !== side) {
          addMove(moves, board, idx, to);
        }
      }
    } else if (type === 'A') {
      for (const dir of ADVISOR_DIRS) {
        const nx = x + dir.dx;
        const ny = y + dir.dy;
        if (!inBounds(nx, ny) || !inPalace(nx, ny, side)) continue;
        const to = coordToIndex(nx, ny);
        if (!board[to] || sideOf(board[to]) !== side) {
          addMove(moves, board, idx, to);
        }
      }
    } else if (type === 'K') {
      for (const dir of KING_DIRS) {
        const nx = x + dir.dx;
        const ny = y + dir.dy;
        if (!inBounds(nx, ny) || !inPalace(nx, ny, side)) continue;
        const to = coordToIndex(nx, ny);
        if (!board[to] || sideOf(board[to]) !== side) {
          addMove(moves, board, idx, to);
        }
      }

      // Flying general capture
      const step = side === 'r' ? -1 : 1;
      let ny = y + step;
      while (inBounds(x, ny)) {
        const to = coordToIndex(x, ny);
        const target = board[to];
        if (target) {
          if (pieceType(target) === 'K' && sideOf(target) !== side) {
            addMove(moves, board, idx, to);
          }
          break;
        }
        ny += step;
      }
    } else if (type === 'P') {
      const forward = side === 'r' ? -1 : 1;
      const nx = x;
      const ny = y + forward;
      if (inBounds(nx, ny)) {
        const to = coordToIndex(nx, ny);
        if (!board[to] || sideOf(board[to]) !== side) {
          addMove(moves, board, idx, to);
        }
      }

      const crossed = side === 'r' ? y <= 4 : y >= 5;
      if (crossed) {
        for (const dx of [-1, 1]) {
          const cx = x + dx;
          const cy = y;
          if (!inBounds(cx, cy)) continue;
          const to = coordToIndex(cx, cy);
          if (!board[to] || sideOf(board[to]) !== side) {
            addMove(moves, board, idx, to);
          }
        }
      }
    }
  }

  return moves;
}

function findKing(board, side) {
  for (let i = 0; i < BOARD_SIZE; i += 1) {
    const piece = board[i];
    if (!piece) continue;
    if (pieceType(piece) === 'K' && sideOf(piece) === side) return i;
  }
  return -1;
}

function isSquareAttacked(board, x, y, attackerSide) {
  // Pawn attacks
  if (attackerSide === 'r') {
    if (inBounds(x, y + 1)) {
      const idx = coordToIndex(x, y + 1);
      if (board[idx] === 'P') return true;
    }
    if (inBounds(x - 1, y)) {
      const idx = coordToIndex(x - 1, y);
      if (board[idx] === 'P' && y <= 4) return true;
    }
    if (inBounds(x + 1, y)) {
      const idx = coordToIndex(x + 1, y);
      if (board[idx] === 'P' && y <= 4) return true;
    }
  } else {
    if (inBounds(x, y - 1)) {
      const idx = coordToIndex(x, y - 1);
      if (board[idx] === 'p') return true;
    }
    if (inBounds(x - 1, y)) {
      const idx = coordToIndex(x - 1, y);
      if (board[idx] === 'p' && y >= 5) return true;
    }
    if (inBounds(x + 1, y)) {
      const idx = coordToIndex(x + 1, y);
      if (board[idx] === 'p' && y >= 5) return true;
    }
  }

  // Horse attacks
  for (const dir of HORSE_DIRS) {
    const hx = x + dir.dx;
    const hy = y + dir.dy;
    if (!inBounds(hx, hy)) continue;
    let lx;
    let ly;
    if (Math.abs(hx - x) === 2) {
      lx = hx - (hx > x ? 1 : -1);
      ly = hy;
    } else {
      lx = hx;
      ly = hy - (hy > y ? 1 : -1);
    }
    if (!inBounds(lx, ly)) continue;
    if (board[coordToIndex(lx, ly)]) continue;
    const piece = board[coordToIndex(hx, hy)];
    if (piece && pieceType(piece) === 'N' && sideOf(piece) === attackerSide) {
      return true;
    }
  }

  // Elephant attacks
  for (const dir of ELE_DIRS) {
    const ex = x + dir.dx;
    const ey = y + dir.dy;
    const bx = x + dir.ex;
    const by = y + dir.ey;
    if (!inBounds(ex, ey) || !inBounds(bx, by)) continue;
    if (board[coordToIndex(bx, by)]) continue;
    const piece = board[coordToIndex(ex, ey)];
    if (!piece || pieceType(piece) !== 'B' || sideOf(piece) !== attackerSide) continue;
    if (attackerSide === 'r' && ey < 5) continue;
    if (attackerSide === 'b' && ey > 4) continue;
    return true;
  }

  // Advisor attacks
  for (const dir of ADVISOR_DIRS) {
    const ax = x + dir.dx;
    const ay = y + dir.dy;
    if (!inBounds(ax, ay) || !inPalace(ax, ay, attackerSide)) continue;
    const piece = board[coordToIndex(ax, ay)];
    if (piece && pieceType(piece) === 'A' && sideOf(piece) === attackerSide) {
      return true;
    }
  }

  // King adjacency
  for (const dir of KING_DIRS) {
    const kx = x + dir.dx;
    const ky = y + dir.dy;
    if (!inBounds(kx, ky)) continue;
    const piece = board[coordToIndex(kx, ky)];
    if (piece && pieceType(piece) === 'K' && sideOf(piece) === attackerSide) {
      return true;
    }
  }

  // Rook, cannon, flying king along ranks/files
  for (const dir of ROOK_DIRS) {
    let nx = x + dir.dx;
    let ny = y + dir.dy;
    let screen = 0;
    while (inBounds(nx, ny)) {
      const idx = coordToIndex(nx, ny);
      const piece = board[idx];
      if (piece) {
        if (screen === 0) {
          if (sideOf(piece) === attackerSide) {
            const t = pieceType(piece);
            if (t === 'R') return true;
            if (t === 'K' && dir.dx === 0) return true;
          }
          screen = 1;
        } else {
          if (sideOf(piece) === attackerSide && pieceType(piece) === 'C') {
            return true;
          }
          break;
        }
      }
      nx += dir.dx;
      ny += dir.dy;
    }
  }

  return false;
}

function findAttackers(board, x, y, attackerSide) {
  const attackers = [];
  const add = (idx) => {
    if (idx === -1) return;
    if (!attackers.includes(idx)) attackers.push(idx);
  };

  // Pawn attacks
  if (attackerSide === 'r') {
    if (inBounds(x, y + 1)) {
      const idx = coordToIndex(x, y + 1);
      if (board[idx] === 'P') add(idx);
    }
    if (inBounds(x - 1, y)) {
      const idx = coordToIndex(x - 1, y);
      if (board[idx] === 'P' && y <= 4) add(idx);
    }
    if (inBounds(x + 1, y)) {
      const idx = coordToIndex(x + 1, y);
      if (board[idx] === 'P' && y <= 4) add(idx);
    }
  } else {
    if (inBounds(x, y - 1)) {
      const idx = coordToIndex(x, y - 1);
      if (board[idx] === 'p') add(idx);
    }
    if (inBounds(x - 1, y)) {
      const idx = coordToIndex(x - 1, y);
      if (board[idx] === 'p' && y >= 5) add(idx);
    }
    if (inBounds(x + 1, y)) {
      const idx = coordToIndex(x + 1, y);
      if (board[idx] === 'p' && y >= 5) add(idx);
    }
  }

  // Horse attacks
  for (const dir of HORSE_DIRS) {
    const hx = x + dir.dx;
    const hy = y + dir.dy;
    if (!inBounds(hx, hy)) continue;
    let lx;
    let ly;
    if (Math.abs(hx - x) === 2) {
      lx = hx - (hx > x ? 1 : -1);
      ly = hy;
    } else {
      lx = hx;
      ly = hy - (hy > y ? 1 : -1);
    }
    if (!inBounds(lx, ly)) continue;
    if (board[coordToIndex(lx, ly)]) continue;
    const idx = coordToIndex(hx, hy);
    const piece = board[idx];
    if (piece && pieceType(piece) === 'N' && sideOf(piece) === attackerSide) {
      add(idx);
    }
  }

  // Elephant attacks
  for (const dir of ELE_DIRS) {
    const ex = x + dir.dx;
    const ey = y + dir.dy;
    const bx = x + dir.ex;
    const by = y + dir.ey;
    if (!inBounds(ex, ey) || !inBounds(bx, by)) continue;
    if (board[coordToIndex(bx, by)]) continue;
    const idx = coordToIndex(ex, ey);
    const piece = board[idx];
    if (!piece || pieceType(piece) !== 'B' || sideOf(piece) !== attackerSide) continue;
    if (attackerSide === 'r' && ey < 5) continue;
    if (attackerSide === 'b' && ey > 4) continue;
    add(idx);
  }

  // Advisor attacks
  for (const dir of ADVISOR_DIRS) {
    const ax = x + dir.dx;
    const ay = y + dir.dy;
    if (!inBounds(ax, ay) || !inPalace(ax, ay, attackerSide)) continue;
    const idx = coordToIndex(ax, ay);
    const piece = board[idx];
    if (piece && pieceType(piece) === 'A' && sideOf(piece) === attackerSide) {
      add(idx);
    }
  }

  // King adjacency
  for (const dir of KING_DIRS) {
    const kx = x + dir.dx;
    const ky = y + dir.dy;
    if (!inBounds(kx, ky)) continue;
    const idx = coordToIndex(kx, ky);
    const piece = board[idx];
    if (piece && pieceType(piece) === 'K' && sideOf(piece) === attackerSide) {
      add(idx);
    }
  }

  // Rook, cannon, flying king along ranks/files
  for (const dir of ROOK_DIRS) {
    let nx = x + dir.dx;
    let ny = y + dir.dy;
    let screen = 0;
    while (inBounds(nx, ny)) {
      const idx = coordToIndex(nx, ny);
      const piece = board[idx];
      if (piece) {
        if (screen == 0) {
          if (sideOf(piece) === attackerSide) {
            const t = pieceType(piece);
            if (t === 'R') add(idx);
            if (t === 'K' && dir.dx === 0) add(idx);
          }
          screen = 1;
        } else {
          if (sideOf(piece) === attackerSide && pieceType(piece) === 'C') {
            add(idx);
          }
          break;
        }
      }
      nx += dir.dx;
      ny += dir.dy;
    }
  }

  return attackers;
}

function findCheckers(board, side) {
  const kingIdx = findKing(board, side);
  if (kingIdx === -1) return [];
  const { x, y } = indexToCoord(kingIdx);
  return findAttackers(board, x, y, opposite(side));
}

function isInCheck(board, side) {
  const kingIdx = findKing(board, side);
  if (kingIdx === -1) return true;
  const { x, y } = indexToCoord(kingIdx);
  return isSquareAttacked(board, x, y, opposite(side));
}

function hasPawnAt(board, x, y, side) {
  if (!inBounds(x, y)) return false;
  const piece = board[coordToIndex(x, y)];
  return piece && pieceType(piece) === 'P' && sideOf(piece) === side;
}

function hasEnemyPawnAhead(board, x, y, side) {
  const enemy = opposite(side);
  if (side === 'r') {
    for (let ny = y - 1; ny >= 0; ny -= 1) {
      if (hasPawnAt(board, x, ny, enemy)) return true;
    }
  } else {
    for (let ny = y + 1; ny < RANKS; ny += 1) {
      if (hasPawnAt(board, x, ny, enemy)) return true;
    }
  }
  return false;
}

function palaceAttackCount(board, attackerSide) {
  const targetSide = opposite(attackerSide);
  const minY = targetSide === 'r' ? 7 : 0;
  const maxY = targetSide === 'r' ? 9 : 2;
  let count = 0;
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = 3; x <= 5; x += 1) {
      if (isSquareAttacked(board, x, y, attackerSide)) count += 1;
    }
  }
  return count;
}

function generateLegalMoves(board, side) {
  if (findKing(board, side) === -1) return [];
  const moves = generatePseudoMoves(board, side);
  const legal = [];
  for (const move of moves) {
    const next = makeMove(board, move);
    if (!isInCheck(next, side)) {
      legal.push(move);
    }
  }
  return legal;
}

function makeMove(board, move) {
  const next = board.slice();
  next[move.to] = next[move.from];
  next[move.from] = null;
  return next;
}


function evaluate(board, weights) {
  let score = 0;
  const redPawnFiles = Array(FILES).fill(0);
  const blackPawnFiles = Array(FILES).fill(0);
  const redPawns = [];
  const blackPawns = [];
  const redRooks = [];
  const blackRooks = [];
  const redCannons = [];
  const blackCannons = [];

  for (let i = 0; i < BOARD_SIZE; i += 1) {
    const piece = board[i];
    if (!piece) continue;
    const type = pieceType(piece);
    const idx = isRed(piece) ? i : mirrorIndex(i);
    const value = weights.values[type] + weights.pst[type][idx];
    if (isRed(piece)) score += value;
    else score -= value;

    const { x, y } = indexToCoord(i);
    if (type === 'P') {
      if (isRed(piece)) {
        redPawns.push({ x, y });
        redPawnFiles[x] += 1;
      } else {
        blackPawns.push({ x, y });
        blackPawnFiles[x] += 1;
      }
    } else if (type === 'R') {
      if (isRed(piece)) redRooks.push({ x, y });
      else blackRooks.push({ x, y });
    } else if (type === 'C') {
      if (isRed(piece)) redCannons.push({ x, y });
      else blackCannons.push({ x, y });
    }
  }

  for (const pawn of redPawns) {
    if (pawn.y <= 4) {
      score += EVAL.PAWN_CROSSED + EVAL.PAWN_ADVANCE * (4 - pawn.y);
    }
    const connected =
      hasPawnAt(board, pawn.x - 1, pawn.y, 'r') ||
      hasPawnAt(board, pawn.x + 1, pawn.y, 'r');
    score += connected ? EVAL.CONNECTED_PAWN : -EVAL.ISOLATED_PAWN;
    if (!hasEnemyPawnAhead(board, pawn.x, pawn.y, 'r')) {
      score += EVAL.PASSED_PAWN;
    }
  }

  for (const pawn of blackPawns) {
    if (pawn.y >= 5) {
      score -= EVAL.PAWN_CROSSED + EVAL.PAWN_ADVANCE * (pawn.y - 5);
    }
    const connected =
      hasPawnAt(board, pawn.x - 1, pawn.y, 'b') ||
      hasPawnAt(board, pawn.x + 1, pawn.y, 'b');
    score -= connected ? EVAL.CONNECTED_PAWN : -EVAL.ISOLATED_PAWN;
    if (!hasEnemyPawnAhead(board, pawn.x, pawn.y, 'b')) {
      score -= EVAL.PASSED_PAWN;
    }
  }

  for (const piece of redRooks.concat(redCannons)) {
    if (redPawnFiles[piece.x] === 0) {
      score += EVAL.OPEN_FILE;
      if (blackPawnFiles[piece.x] === 0) score += EVAL.OPEN_FILE_FULL;
    }
  }

  for (const piece of blackRooks.concat(blackCannons)) {
    if (blackPawnFiles[piece.x] === 0) {
      score -= EVAL.OPEN_FILE;
      if (redPawnFiles[piece.x] === 0) score -= EVAL.OPEN_FILE_FULL;
    }
  }

  const redMobility = generatePseudoMoves(board, 'r').length;
  const blackMobility = generatePseudoMoves(board, 'b').length;
  score += EVAL.MOBILITY_WEIGHT * (redMobility - blackMobility);

  if (isInCheck(board, 'r')) score -= EVAL.CHECK_PENALTY;
  if (isInCheck(board, 'b')) score += EVAL.CHECK_PENALTY;

  const redPressure = palaceAttackCount(board, 'r');
  const blackPressure = palaceAttackCount(board, 'b');
  score += EVAL.PALACE_PRESSURE * (redPressure - blackPressure);

  return score;
}

function gameStatus(board, side) {
  const redKing = findKing(board, 'r');
  const blackKing = findKing(board, 'b');
  if (redKing === -1 && blackKing === -1) {
    return { over: true, winner: null, reason: 'no_kings' };
  }
  if (redKing === -1) {
    return { over: true, winner: 'b', reason: 'king_captured' };
  }
  if (blackKing === -1) {
    return { over: true, winner: 'r', reason: 'king_captured' };
  }

  const moves = generateLegalMoves(board, side);
  if (moves.length > 0) {
    return { over: false, winner: null, reason: null };
  }
  if (isInCheck(board, side)) {
    return { over: true, winner: opposite(side), reason: 'checkmate' };
  }
  // Xiangqi rule: no legal moves is a loss (even if not in check)
  return { over: true, winner: opposite(side), reason: 'no_moves' };
}

function negamax(board, side, depth, alpha, beta, weights, tt, searchState) {
  if (searchState && searchState.timeLimitMs) {
    if (nowMs() - searchState.start >= searchState.timeLimitMs) {
      searchState.stop = true;
    }
  }
  if (searchState && searchState.stop) {
    return evaluate(board, weights) * (side === 'r' ? 1 : -1);
  }

  const moves = generateLegalMoves(board, side);
  if (moves.length === 0) {
    if (isInCheck(board, side)) {
      return -MATE_SCORE + depth;
    }
    return 0;
  }
  if (depth === 0) {
    return evaluate(board, weights) * (side === 'r' ? 1 : -1);
  }

  const key = tt ? positionKey(board, side) : null;
  const entry = tt && key ? tt.get(key) : null;
  let alphaOrig = alpha;
  let betaOrig = beta;

  if (entry && entry.depth >= depth) {
    if (entry.flag === 'EXACT') return entry.score;
    if (entry.flag === 'LOWER') alpha = Math.max(alpha, entry.score);
    else if (entry.flag === 'UPPER') beta = Math.min(beta, entry.score);
    if (alpha >= beta) return entry.score;
  }

  orderMoves(moves, entry ? entry.bestMove : null, weights);

  let best = -Infinity;
  let bestMove = null;

  for (const move of moves) {
    if (searchState && searchState.stop) break;
    const next = makeMove(board, move);
    const score = -negamax(next, opposite(side), depth - 1, -beta, -alpha, weights, tt, searchState);
    if (score > best) {
      best = score;
      bestMove = move;
    }
    if (score > alpha) alpha = score;
    if (alpha >= beta) break;
    if (searchState && searchState.timeLimitMs && nowMs() - searchState.start >= searchState.timeLimitMs) {
      searchState.stop = true;
      break;
    }
  }

  if (tt && key) {
    let flag = 'EXACT';
    if (best <= alphaOrig) flag = 'UPPER';
    else if (best >= betaOrig) flag = 'LOWER';
    tt.set(key, { depth, score: best, flag, bestMove });
    if (tt.size > (searchState?.ttMax || 60000)) tt.clear();
  }

  return best;
}

function rootSearch(board, side, weights, rootMoves, maxDepth, timeLimitMs, ttMax) {
  const moves = rootMoves;
  const tt = new Map();
  const searchState = {
    start: nowMs(),
    timeLimitMs,
    stop: false,
    ttMax: ttMax || 60000,
  };

  let bestMove = null;
  let bestScore = -Infinity;
  let reachedDepth = 0;

  for (let d = 1; d <= maxDepth; d += 1) {
    if (searchState.stop) break;
    orderMoves(moves, bestMove, weights);

    let alpha = -Infinity;
    let beta = Infinity;
    let localBest = -Infinity;
    let localMove = null;

    for (const move of moves) {
      if (timeLimitMs && nowMs() - searchState.start >= timeLimitMs) {
        searchState.stop = true;
        break;
      }
      const next = makeMove(board, move);
      const score = -negamax(next, opposite(side), d - 1, -beta, -alpha, weights, tt, searchState);
      if (score > localBest) {
        localBest = score;
        localMove = move;
      }
      if (score > alpha) alpha = score;
      if (alpha >= beta) break;
      if (searchState.stop) break;
    }

    if (!searchState.stop && localMove) {
      bestMove = localMove;
      bestScore = localBest;
      reachedDepth = d;
    }
  }

  return { move: bestMove, score: bestScore, depth: reachedDepth };
}

function searchBestMove(board, side, weights, depth, options = {}) {

  const maxDepth = options.maxDepth || depth;
  const timeLimitMs = options.timeLimitMs || 0;
  const moves = generateLegalMoves(board, side);
  if (moves.length === 0) return { move: null, score: -MATE_SCORE, depth: 0 };

  return rootSearch(board, side, weights, moves, maxDepth, timeLimitMs, options.ttMax);
}

function searchBestMoveFromMoves(board, side, weights, depth, options = {}) {
  const maxDepth = options.maxDepth || depth;
  const timeLimitMs = options.timeLimitMs || 0;
  const moves = Array.isArray(options.moves) ? options.moves.slice() : generateLegalMoves(board, side);
  if (moves.length === 0) return { move: null, score: -MATE_SCORE, depth: 0 };

  return rootSearch(board, side, weights, moves, maxDepth, timeLimitMs, options.ttMax);
}

function updateWeightsFromHistory(weights, history, result, lr) {
  if (result === 0) return;
  const step = lr / Math.max(1, history.length);
  for (const board of history) {
    for (let i = 0; i < BOARD_SIZE; i += 1) {
      const piece = board[i];
      if (!piece) continue;
      const type = pieceType(piece);
      if (type === 'K') continue;
      const sign = isRed(piece) ? 1 : -1;
      weights.values[type] += step * result * sign;
      const idx = isRed(piece) ? i : mirrorIndex(i);
      weights.pst[type][idx] += step * result * sign;
    }
  }
  clampWeights(weights);
}

export {
  FILES,
  RANKS,
  BOARD_SIZE,
  coordToIndex,
  indexToCoord,
  inBounds,
  initialBoard,
  defaultWeights,
  cloneWeights,
  mirrorIndex,
  positionKey,
  isRed,
  isBlack,
  pieceType,
  opposite,
  isInCheck,
  findCheckers,
  generatePseudoMoves,
  generateLegalMoves,
  makeMove,
  evaluate,
  gameStatus,
  searchBestMove,
  searchBestMoveFromMoves,
  updateWeightsFromHistory,
};
