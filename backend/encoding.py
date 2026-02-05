from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from .engine import BOARD_SIZE, Move, index_to_coord

PIECE_PLANES = ["K", "A", "B", "N", "R", "C", "P", "k", "a", "b", "n", "r", "c", "p"]
PIECE_TO_PLANE = {piece: idx for idx, piece in enumerate(PIECE_PLANES)}
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE


def encode_board(board, side: str) -> torch.Tensor:
    planes = np.zeros((15, 10, 9), dtype=np.float32)
    for idx, piece in enumerate(board):
        if not piece:
            continue
        plane = PIECE_TO_PLANE[piece]
        x, y = index_to_coord(idx)
        planes[plane, y, x] = 1.0
    planes[14, :, :] = 1.0 if side == "r" else -1.0
    return torch.from_numpy(planes)


def move_to_action(move: Move) -> int:
    return move.from_idx * 90 + move.to_idx


def action_to_move(action: int) -> Move:
    from_idx = action // 90
    to_idx = action % 90
    return Move(from_idx, to_idx)


def visits_to_policy(visit_counts: Dict[int, int]) -> np.ndarray:
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    total = sum(visit_counts.values())
    if total <= 0:
        return policy
    for action, count in visit_counts.items():
        policy[action] = count / total
    return policy
