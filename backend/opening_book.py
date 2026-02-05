from __future__ import annotations

import random
from typing import Dict, List, Optional

from .engine import (
    Board,
    Move,
    coord_to_index,
    generate_legal_moves,
    initial_board,
    make_move,
    position_key,
)


def _move(fx: int, fy: int, tx: int, ty: int) -> Move:
    return Move(coord_to_index(fx, fy), coord_to_index(tx, ty))


def _seq(board: Board, moves: List[Move]) -> Board:
    next_board = board
    for mv in moves:
        next_board = make_move(next_board, mv)
    return next_board


def _add_entry(book: Dict[str, List[Move]], board: Board, side: str, moves: List[Move]) -> None:
    book[position_key(board, side)] = moves


def build_opening_book() -> Dict[str, List[Move]]:
    book: Dict[str, List[Move]] = {}

    start = initial_board()
    red_c2 = _move(1, 7, 4, 7)
    red_c8 = _move(7, 7, 4, 7)
    red_h2 = _move(1, 9, 2, 7)
    red_h8 = _move(7, 9, 6, 7)
    red_p5 = _move(4, 6, 4, 5)
    red_p7 = _move(2, 6, 2, 5)
    red_p3 = _move(6, 6, 6, 5)
    black_c8 = _move(7, 2, 4, 2)
    black_c2 = _move(1, 2, 4, 2)
    black_h8 = _move(7, 0, 6, 2)
    black_h2 = _move(1, 0, 2, 2)
    black_a3 = _move(3, 0, 4, 1)
    black_a5 = _move(5, 0, 4, 1)

    _add_entry(book, start, "r", [red_c2, red_c8, red_h2, red_h8, red_p5, red_p7, red_p3])

    board_p5 = make_move(start, red_p5)
    _add_entry(book, board_p5, "b", [_move(4, 3, 4, 4), black_h8, black_h2])

    board_p7 = make_move(start, red_p7)
    _add_entry(book, board_p7, "b", [black_h2, black_h8, black_c2])

    board_p3 = make_move(start, red_p3)
    _add_entry(book, board_p3, "b", [black_h8, black_h2, black_c8])

    # 中炮开局
    board_c2 = make_move(start, red_c2)
    _add_entry(book, board_c2, "b", [black_c8, black_h8, black_h2, black_a3, black_a5])

    board_c2b = make_move(board_c2, black_c8)
    _add_entry(book, board_c2b, "r", [red_h2, red_h8, red_p5])

    # 屏风马分支
    board_c2_h8 = make_move(board_c2, black_h8)
    _add_entry(book, board_c2_h8, "r", [red_h2, red_h8, red_p5])

    board_c2_h8_r2 = make_move(board_c2_h8, red_h2)
    _add_entry(book, board_c2_h8_r2, "b", [black_h2, black_c8, black_a3])

    board_c2_h8_r2_h2 = make_move(board_c2_h8_r2, black_h2)
    _add_entry(book, board_c2_h8_r2_h2, "r", [red_h8, red_p5, _move(3, 6, 3, 5)])

    board_c2_h8_r8 = make_move(board_c2_h8, red_h8)
    _add_entry(book, board_c2_h8_r8, "b", [black_h2, black_c8, black_a5])

    board_c2_h8_r8_h2 = make_move(board_c2_h8_r8, black_h2)
    _add_entry(book, board_c2_h8_r8_h2, "r", [red_h2, red_p5, _move(5, 6, 5, 5)])

    # 反宫马分支
    board_c2_h2 = make_move(board_c2, black_h2)
    _add_entry(book, board_c2_h2, "r", [red_h8, red_h2, red_p5])

    board_c2_h2_r2 = make_move(board_c2_h2, red_h2)
    _add_entry(book, board_c2_h2_r2, "b", [black_h8, black_c8, black_a3, black_a5])

    board_c2_h2_r2_h8 = make_move(board_c2_h2_r2, black_h8)
    _add_entry(book, board_c2_h2_r2_h8, "r", [red_h8, red_p5, _move(3, 6, 3, 5)])

    board_c2_h2_r8 = make_move(board_c2_h2, red_h8)
    _add_entry(book, board_c2_h2_r8, "b", [black_h8, black_c8, black_a5])

    board_c2_h2_r8_h8 = make_move(board_c2_h2_r8, black_h8)
    _add_entry(book, board_c2_h2_r8_h8, "r", [red_h2, red_p5, _move(5, 6, 5, 5)])

    # 顺炮开局
    board_c8 = make_move(start, red_c8)
    _add_entry(book, board_c8, "b", [black_c2, black_h2, black_h8, black_a3, black_a5])

    board_c8b = make_move(board_c8, black_c2)
    _add_entry(book, board_c8b, "r", [red_h8, red_h2, red_p5])

    board_c8_h2 = make_move(board_c8, black_h2)
    _add_entry(book, board_c8_h2, "r", [red_h8, red_h2, red_p5])

    board_c8_h2_r8 = make_move(board_c8_h2, red_h8)
    _add_entry(book, board_c8_h2_r8, "b", [black_h8, black_c2, black_a5])

    board_c8_h8 = make_move(board_c8, black_h8)
    _add_entry(book, board_c8_h8, "r", [red_h2, red_h8, red_p5])

    board_c8_h8_r2 = make_move(board_c8_h8, red_h2)
    _add_entry(book, board_c8_h8_r2, "b", [black_h2, black_c2, black_a3])

    # 先马开局补充
    board_h2 = make_move(start, red_h2)
    _add_entry(book, board_h2, "b", [black_h8, black_c8, black_c2])

    board_h2b = make_move(board_h2, black_h8)
    _add_entry(book, board_h2b, "r", [red_c2, red_c8, red_p5])

    board_h8 = make_move(start, red_h8)
    _add_entry(book, board_h8, "b", [black_h2, black_c2, black_c8])

    board_h8b = make_move(board_h8, black_h2)
    _add_entry(book, board_h8b, "r", [red_c8, red_c2, red_p5])

    # 中炮后对方上士
    board_c2_a3 = _seq(start, [red_c2, black_a3])
    _add_entry(book, board_c2_a3, "r", [red_h2, red_h8, red_p5])

    board_c2_a5 = _seq(start, [red_c2, black_a5])
    _add_entry(book, board_c2_a5, "r", [red_h8, red_h2, red_p5])

    return book


OPENING_BOOK = build_opening_book()


def opening_book_move(board: Board, side: str) -> Optional[Move]:
    entries = OPENING_BOOK.get(position_key(board, side))
    if not entries:
        return None
    legal = generate_legal_moves(board, side)
    legal_map = {(mv.from_idx, mv.to_idx): mv for mv in legal}
    candidates = [legal_map.get((mv.from_idx, mv.to_idx)) for mv in entries]
    candidates = [mv for mv in candidates if mv]
    if not candidates:
        return None
    return random.choice(candidates)
