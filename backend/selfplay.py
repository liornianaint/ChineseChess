from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from .engine import Board, game_status, initial_board, make_move, opposite, position_key
from .encoding import action_to_move, visits_to_policy
from .mcts import MCTSConfig, run_mcts, select_action, visit_counts


@dataclass
class SelfPlayConfig:
    simulations: int = 320
    max_moves: int = 220
    temperature_moves: int = 12
    mcts_batch_size: int = 64


TrainingExample = Tuple[Board, str, np.ndarray, float]


def self_play_game(
    model,
    config: SelfPlayConfig,
    device: torch.device,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> Tuple[List[TrainingExample], int, float]:
    board = initial_board()
    side = "r"
    history: List[Tuple[Board, str, np.ndarray]] = []
    position_counts = {}
    cache = {}
    root = None
    mcts = MCTSConfig(simulations=config.simulations, batch_size=config.mcts_batch_size)

    for move_idx in range(config.max_moves):
        key = position_key(board, side)
        position_counts[key] = position_counts.get(key, 0) + 1
        if position_counts[key] >= 3:
            result = 0
            break

        root = run_mcts(model, board, side, mcts, device, root=root, cache=cache)
        visits = visit_counts(root)
        policy = visits_to_policy(visits).astype(np.float16)
        temperature = 1.0 if move_idx < config.temperature_moves else 0.1
        action = select_action(root, temperature)
        if action is None:
            result = 0
            break

        next_root = None
        if root and action is not None:
            next_root = root.children.get(action)

        history.append((board, side, policy))
        if progress_cb:
            progress_cb(1)
        move = action_to_move(action)
        board = make_move(board, move)
        side = opposite(side)
        root = next_root

        over, winner = game_status(board, side)
        if over:
            if winner is None:
                result = 0
            else:
                result = 1 if winner == "r" else -1
            break
    else:
        result = 0

    examples: List[TrainingExample] = []
    for state, state_side, policy in history:
        value = result if state_side == "r" else -result
        examples.append((state, state_side, policy, float(value)))

    return examples, len(history), float(result)
