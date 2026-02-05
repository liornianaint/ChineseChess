from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

FILES = 9
RANKS = 10
BOARD_SIZE = FILES * RANKS

PIECE_TYPES = ["K", "A", "B", "N", "R", "C", "P"]

HORSE_DIRS = [
    (2, 1, 1, 0),
    (2, -1, 1, 0),
    (-2, 1, -1, 0),
    (-2, -1, -1, 0),
    (1, 2, 0, 1),
    (-1, 2, 0, 1),
    (1, -2, 0, -1),
    (-1, -2, 0, -1),
]

ELE_DIRS = [
    (2, 2, 1, 1),
    (2, -2, 1, -1),
    (-2, 2, -1, 1),
    (-2, -2, -1, -1),
]

ADVISOR_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
KING_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
ROOK_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

Board = List[Optional[str]]


@dataclass(frozen=True)
class Move:
    from_idx: int
    to_idx: int


def coord_to_index(x: int, y: int) -> int:
    return y * FILES + x


def index_to_coord(idx: int) -> Tuple[int, int]:
    return idx % FILES, idx // FILES


def in_bounds(x: int, y: int) -> bool:
    return 0 <= x < FILES and 0 <= y < RANKS


def is_red(piece: Optional[str]) -> bool:
    return bool(piece) and piece.isupper()


def is_black(piece: Optional[str]) -> bool:
    return bool(piece) and piece.islower()


def piece_type(piece: str) -> str:
    return piece.upper()


def side_of(piece: str) -> str:
    return "r" if is_red(piece) else "b"


def opposite(side: str) -> str:
    return "b" if side == "r" else "r"


def mirror_index(idx: int) -> int:
    x, y = index_to_coord(idx)
    return coord_to_index(x, RANKS - 1 - y)


def in_palace(x: int, y: int, side: str) -> bool:
    if x < 3 or x > 5:
        return False
    if side == "r":
        return 7 <= y <= 9
    return 0 <= y <= 2


def initial_board() -> Board:
    board: Board = [None] * BOARD_SIZE

    def place_row(y: int, row: List[Optional[str]]) -> None:
        for x, piece in enumerate(row):
            if piece:
                board[coord_to_index(x, y)] = piece

    place_row(0, ["r", "n", "b", "a", "k", "a", "b", "n", "r"])
    place_row(1, [None] * 9)
    place_row(2, [None, "c", None, None, None, None, None, "c", None])
    place_row(3, ["p", None, "p", None, "p", None, "p", None, "p"])
    place_row(4, [None] * 9)
    place_row(5, [None] * 9)
    place_row(6, ["P", None, "P", None, "P", None, "P", None, "P"])
    place_row(7, [None, "C", None, None, None, None, None, "C", None])
    place_row(8, [None] * 9)
    place_row(9, ["R", "N", "B", "A", "K", "A", "B", "N", "R"])

    return board


def position_key(board: Board, side: str) -> str:
    return f"{side}:" + "".join(piece if piece else "." for piece in board)


def add_move(moves: List[Move], board: Board, from_idx: int, to_idx: int) -> None:
    target = board[to_idx]
    if target and side_of(target) == side_of(board[from_idx]):
        return
    moves.append(Move(from_idx, to_idx))


def generate_pseudo_moves(board: Board, side: str) -> List[Move]:
    moves: List[Move] = []

    for idx, piece in enumerate(board):
        if not piece:
            continue
        if side == "r" and not is_red(piece):
            continue
        if side == "b" and not is_black(piece):
            continue

        x, y = index_to_coord(idx)
        ptype = piece_type(piece)

        if ptype == "R":
            for dx, dy in ROOK_DIRS:
                nx = x + dx
                ny = y + dy
                while in_bounds(nx, ny):
                    to_idx = coord_to_index(nx, ny)
                    if not board[to_idx]:
                        add_move(moves, board, idx, to_idx)
                    else:
                        if side_of(board[to_idx]) != side:
                            add_move(moves, board, idx, to_idx)
                        break
                    nx += dx
                    ny += dy
        elif ptype == "C":
            for dx, dy in ROOK_DIRS:
                nx = x + dx
                ny = y + dy
                screen = False
                while in_bounds(nx, ny):
                    to_idx = coord_to_index(nx, ny)
                    target = board[to_idx]
                    if not screen:
                        if not target:
                            add_move(moves, board, idx, to_idx)
                        else:
                            screen = True
                    else:
                        if target:
                            if side_of(target) != side:
                                add_move(moves, board, idx, to_idx)
                            break
                    nx += dx
                    ny += dy
        elif ptype == "N":
            for dx, dy, lx, ly in HORSE_DIRS:
                leg_x = x + lx
                leg_y = y + ly
                if not in_bounds(leg_x, leg_y):
                    continue
                if board[coord_to_index(leg_x, leg_y)]:
                    continue
                nx = x + dx
                ny = y + dy
                if not in_bounds(nx, ny):
                    continue
                to_idx = coord_to_index(nx, ny)
                if not board[to_idx] or side_of(board[to_idx]) != side:
                    add_move(moves, board, idx, to_idx)
        elif ptype == "B":
            for dx, dy, ex, ey in ELE_DIRS:
                nx = x + dx
                ny = y + dy
                block_x = x + ex
                block_y = y + ey
                if not in_bounds(nx, ny) or not in_bounds(block_x, block_y):
                    continue
                if board[coord_to_index(block_x, block_y)]:
                    continue
                if side == "r" and ny < 5:
                    continue
                if side == "b" and ny > 4:
                    continue
                to_idx = coord_to_index(nx, ny)
                if not board[to_idx] or side_of(board[to_idx]) != side:
                    add_move(moves, board, idx, to_idx)
        elif ptype == "A":
            for dx, dy in ADVISOR_DIRS:
                nx = x + dx
                ny = y + dy
                if not in_bounds(nx, ny) or not in_palace(nx, ny, side):
                    continue
                to_idx = coord_to_index(nx, ny)
                if not board[to_idx] or side_of(board[to_idx]) != side:
                    add_move(moves, board, idx, to_idx)
        elif ptype == "K":
            for dx, dy in KING_DIRS:
                nx = x + dx
                ny = y + dy
                if not in_bounds(nx, ny) or not in_palace(nx, ny, side):
                    continue
                to_idx = coord_to_index(nx, ny)
                if not board[to_idx] or side_of(board[to_idx]) != side:
                    add_move(moves, board, idx, to_idx)

            step = -1 if side == "r" else 1
            ny = y + step
            while in_bounds(x, ny):
                to_idx = coord_to_index(x, ny)
                target = board[to_idx]
                if target:
                    if piece_type(target) == "K" and side_of(target) != side:
                        add_move(moves, board, idx, to_idx)
                    break
                ny += step
        elif ptype == "P":
            forward = -1 if side == "r" else 1
            nx = x
            ny = y + forward
            if in_bounds(nx, ny):
                to_idx = coord_to_index(nx, ny)
                if not board[to_idx] or side_of(board[to_idx]) != side:
                    add_move(moves, board, idx, to_idx)

            crossed = y <= 4 if side == "r" else y >= 5
            if crossed:
                for dx in (-1, 1):
                    cx = x + dx
                    cy = y
                    if not in_bounds(cx, cy):
                        continue
                    to_idx = coord_to_index(cx, cy)
                    if not board[to_idx] or side_of(board[to_idx]) != side:
                        add_move(moves, board, idx, to_idx)

    return moves


def find_king(board: Board, side: str) -> int:
    for idx, piece in enumerate(board):
        if not piece:
            continue
        if piece_type(piece) == "K" and side_of(piece) == side:
            return idx
    return -1


def is_square_attacked(board: Board, x: int, y: int, attacker_side: str) -> bool:
    if attacker_side == "r":
        if in_bounds(x, y + 1):
            idx = coord_to_index(x, y + 1)
            if board[idx] == "P":
                return True
        if in_bounds(x - 1, y):
            idx = coord_to_index(x - 1, y)
            if board[idx] == "P" and y <= 4:
                return True
        if in_bounds(x + 1, y):
            idx = coord_to_index(x + 1, y)
            if board[idx] == "P" and y <= 4:
                return True
    else:
        if in_bounds(x, y - 1):
            idx = coord_to_index(x, y - 1)
            if board[idx] == "p":
                return True
        if in_bounds(x - 1, y):
            idx = coord_to_index(x - 1, y)
            if board[idx] == "p" and y >= 5:
                return True
        if in_bounds(x + 1, y):
            idx = coord_to_index(x + 1, y)
            if board[idx] == "p" and y >= 5:
                return True

    for dx, dy, lx, ly in HORSE_DIRS:
        hx = x + dx
        hy = y + dy
        leg_x = x + lx
        leg_y = y + ly
        if not in_bounds(hx, hy) or not in_bounds(leg_x, leg_y):
            continue
        if board[coord_to_index(leg_x, leg_y)]:
            continue
        piece = board[coord_to_index(hx, hy)]
        if piece and piece_type(piece) == "N" and side_of(piece) == attacker_side:
            return True

    for dx, dy, ex, ey in ELE_DIRS:
        exx = x + dx
        eyy = y + dy
        bx = x + ex
        by = y + ey
        if not in_bounds(exx, eyy) or not in_bounds(bx, by):
            continue
        if board[coord_to_index(bx, by)]:
            continue
        piece = board[coord_to_index(exx, eyy)]
        if not piece or piece_type(piece) != "B" or side_of(piece) != attacker_side:
            continue
        if attacker_side == "r" and eyy < 5:
            continue
        if attacker_side == "b" and eyy > 4:
            continue
        return True

    for dx, dy in ADVISOR_DIRS:
        ax = x + dx
        ay = y + dy
        if not in_bounds(ax, ay) or not in_palace(ax, ay, attacker_side):
            continue
        piece = board[coord_to_index(ax, ay)]
        if piece and piece_type(piece) == "A" and side_of(piece) == attacker_side:
            return True

    for dx, dy in KING_DIRS:
        kx = x + dx
        ky = y + dy
        if not in_bounds(kx, ky):
            continue
        piece = board[coord_to_index(kx, ky)]
        if piece and piece_type(piece) == "K" and side_of(piece) == attacker_side:
            return True

    for dx, dy in ROOK_DIRS:
        nx = x + dx
        ny = y + dy
        screen = 0
        while in_bounds(nx, ny):
            idx = coord_to_index(nx, ny)
            piece = board[idx]
            if piece:
                if screen == 0:
                    if side_of(piece) == attacker_side:
                        ptype = piece_type(piece)
                        if ptype == "R":
                            return True
                        if ptype == "K" and dx == 0:
                            return True
                    screen = 1
                else:
                    if side_of(piece) == attacker_side and piece_type(piece) == "C":
                        return True
                    break
            nx += dx
            ny += dy

    return False


def find_checkers(board: Board, side: str) -> List[int]:
    king_idx = find_king(board, side)
    if king_idx == -1:
        return []
    x, y = index_to_coord(king_idx)
    return find_attackers(board, x, y, opposite(side))


def find_attackers(board: Board, x: int, y: int, attacker_side: str) -> List[int]:
    attackers: List[int] = []

    def add(idx: int) -> None:
        if idx == -1:
            return
        if idx not in attackers:
            attackers.append(idx)

    if attacker_side == "r":
        if in_bounds(x, y + 1):
            idx = coord_to_index(x, y + 1)
            if board[idx] == "P":
                add(idx)
        if in_bounds(x - 1, y):
            idx = coord_to_index(x - 1, y)
            if board[idx] == "P" and y <= 4:
                add(idx)
        if in_bounds(x + 1, y):
            idx = coord_to_index(x + 1, y)
            if board[idx] == "P" and y <= 4:
                add(idx)
    else:
        if in_bounds(x, y - 1):
            idx = coord_to_index(x, y - 1)
            if board[idx] == "p":
                add(idx)
        if in_bounds(x - 1, y):
            idx = coord_to_index(x - 1, y)
            if board[idx] == "p" and y >= 5:
                add(idx)
        if in_bounds(x + 1, y):
            idx = coord_to_index(x + 1, y)
            if board[idx] == "p" and y >= 5:
                add(idx)

    for dx, dy, lx, ly in HORSE_DIRS:
        hx = x + dx
        hy = y + dy
        leg_x = x + lx
        leg_y = y + ly
        if not in_bounds(hx, hy) or not in_bounds(leg_x, leg_y):
            continue
        if board[coord_to_index(leg_x, leg_y)]:
            continue
        idx = coord_to_index(hx, hy)
        piece = board[idx]
        if piece and piece_type(piece) == "N" and side_of(piece) == attacker_side:
            add(idx)

    for dx, dy, ex, ey in ELE_DIRS:
        exx = x + dx
        eyy = y + dy
        bx = x + ex
        by = y + ey
        if not in_bounds(exx, eyy) or not in_bounds(bx, by):
            continue
        if board[coord_to_index(bx, by)]:
            continue
        idx = coord_to_index(exx, eyy)
        piece = board[idx]
        if not piece or piece_type(piece) != "B" or side_of(piece) != attacker_side:
            continue
        if attacker_side == "r" and eyy < 5:
            continue
        if attacker_side == "b" and eyy > 4:
            continue
        add(idx)

    for dx, dy in ADVISOR_DIRS:
        ax = x + dx
        ay = y + dy
        if not in_bounds(ax, ay) or not in_palace(ax, ay, attacker_side):
            continue
        idx = coord_to_index(ax, ay)
        piece = board[idx]
        if piece and piece_type(piece) == "A" and side_of(piece) == attacker_side:
            add(idx)

    for dx, dy in KING_DIRS:
        kx = x + dx
        ky = y + dy
        if not in_bounds(kx, ky):
            continue
        idx = coord_to_index(kx, ky)
        piece = board[idx]
        if piece and piece_type(piece) == "K" and side_of(piece) == attacker_side:
            add(idx)

    for dx, dy in ROOK_DIRS:
        nx = x + dx
        ny = y + dy
        screen = 0
        while in_bounds(nx, ny):
            idx = coord_to_index(nx, ny)
            piece = board[idx]
            if piece:
                if screen == 0:
                    if side_of(piece) == attacker_side:
                        ptype = piece_type(piece)
                        if ptype == "R":
                            add(idx)
                        if ptype == "K" and dx == 0:
                            add(idx)
                    screen = 1
                else:
                    if side_of(piece) == attacker_side and piece_type(piece) == "C":
                        add(idx)
                    break
            nx += dx
            ny += dy

    return attackers


def is_in_check(board: Board, side: str) -> bool:
    king_idx = find_king(board, side)
    if king_idx == -1:
        return True
    x, y = index_to_coord(king_idx)
    return is_square_attacked(board, x, y, opposite(side))


def generate_legal_moves(board: Board, side: str) -> List[Move]:
    if find_king(board, side) == -1:
        return []
    moves = generate_pseudo_moves(board, side)
    legal: List[Move] = []
    for move in moves:
        next_board = make_move(board, move)
        if not is_in_check(next_board, side):
            legal.append(move)
    return legal


def make_move(board: Board, move: Move) -> Board:
    next_board = list(board)
    next_board[move.to_idx] = next_board[move.from_idx]
    next_board[move.from_idx] = None
    return next_board


def game_status(board: Board, side: str) -> Tuple[bool, Optional[str]]:
    red_king = find_king(board, "r")
    black_king = find_king(board, "b")
    if red_king == -1 and black_king == -1:
        return True, None
    if red_king == -1:
        return True, "b"
    if black_king == -1:
        return True, "r"

    moves = generate_legal_moves(board, side)
    if moves:
        return False, None
    if is_in_check(board, side):
        return True, opposite(side)
    return True, opposite(side)
