from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .engine import Board, Move, game_status, generate_legal_moves, make_move, opposite, position_key
from .encoding import action_to_move, encode_board, move_to_action


@dataclass
class MCTSConfig:
    simulations: int = 320
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    batch_size: int = 64


class Node:
    def __init__(self, prior: float):
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, "Node"] = {}

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def evaluate(model, board: Board, side: str, device: torch.device) -> Tuple[np.ndarray, float]:
    model.eval()
    with torch.inference_mode():
        amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with amp_ctx:
            x = encode_board(board, side).unsqueeze(0).to(device)
            logits, value = model(x)
        logits = logits[0].float().cpu().numpy()
        value = float(value[0].item())
    return logits, value


def evaluate_batch(model, boards, sides, device: torch.device):
    model.eval()
    with torch.inference_mode():
        amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with amp_ctx:
            inputs = torch.stack([encode_board(b, s) for b, s in zip(boards, sides)]).to(device)
            logits, values = model(inputs)
        logits = logits.float().cpu().numpy()
        values = values.squeeze(1).float().cpu().numpy()
    return logits, values


def normalize_logits(logits: np.ndarray, legal_actions: List[int]) -> Dict[int, float]:
    if not legal_actions:
        return {}
    legal_logits = np.array([logits[a] for a in legal_actions], dtype=np.float32)
    max_logit = float(np.max(legal_logits)) if len(legal_logits) else 0.0
    exp = np.exp(legal_logits - max_logit)
    denom = float(exp.sum()) + 1e-8
    probs = exp / denom
    return {action: float(prob) for action, prob in zip(legal_actions, probs)}


def add_dirichlet_noise(node: Node, legal_actions: List[int], alpha: float, eps: float) -> None:
    if not legal_actions:
        return
    noise = np.random.dirichlet([alpha] * len(legal_actions))
    for action, n in zip(legal_actions, noise):
        child = node.children.get(action)
        if not child:
            continue
        child.prior = child.prior * (1.0 - eps) + float(n) * eps


def select_child(node: Node, c_puct: float) -> Tuple[int, Node]:
    best_score = -1e9
    best_action = -1
    best_child = None
    sqrt_visits = math.sqrt(node.visit_count + 1e-8)
    for action, child in node.children.items():
        u = c_puct * child.prior * sqrt_visits / (1 + child.visit_count)
        score = child.value + u
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    return best_action, best_child


def run_mcts(
    model,
    board: Board,
    side: str,
    config: MCTSConfig,
    device: torch.device,
    root: Optional[Node] = None,
    cache: Optional[Dict[str, Tuple[np.ndarray, float]]] = None,
) -> Node:
    if cache is None:
        cache = {}

    if root is None:
        root = Node(1.0)
    else:
        root.prior = 1.0

    legal_moves = generate_legal_moves(board, side)
    if not legal_moves:
        return root

    legal_actions = [move_to_action(m) for m in legal_moves]
    if not root.children:
        key = position_key(board, side)
        cached = cache.get(key)
        if cached is None:
            logits, value = evaluate(model, board, side, device)
            cache[key] = (logits, value)
        else:
            logits, value = cached
        priors = normalize_logits(logits, legal_actions)
        root.children = {action: Node(prior) for action, prior in priors.items()}
    add_dirichlet_noise(root, legal_actions, config.dirichlet_alpha, config.dirichlet_eps)

    pending: List[Tuple[Node, Board, str, List[Node], List[int], str]] = []

    def backup(path, value: float) -> None:
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value

    def flush_pending() -> None:
        if not pending:
            return
        boards = [item[1] for item in pending]
        sides = [item[2] for item in pending]
        logits_batch, values_batch = evaluate_batch(model, boards, sides, device)
        for (node, _board, _side, path, actions, key), logits, value in zip(pending, logits_batch, values_batch):
            cache[key] = (logits, float(value))
            priors = normalize_logits(logits, actions)
            node.children = {action: Node(prior) for action, prior in priors.items()}
            backup(path, float(value))
        pending.clear()

    for _ in range(config.simulations):
        node = root
        search_path = [node]
        cur_board = board
        cur_side = side
        terminal = False
        value = 0.0

        while node.children:
            action, child = select_child(node, config.c_puct)
            move = action_to_move(action)
            cur_board = make_move(cur_board, move)
            cur_side = opposite(cur_side)
            node = child
            search_path.append(node)

            over, winner = game_status(cur_board, cur_side)
            if over:
                terminal = True
                if winner is None:
                    value = 0.0
                else:
                    value = 1.0 if winner == cur_side else -1.0
                break

        if terminal:
            backup(search_path, value)
            continue

        over, winner = game_status(cur_board, cur_side)
        if over:
            if winner is None:
                value = 0.0
            else:
                value = 1.0 if winner == cur_side else -1.0
            backup(search_path, value)
            continue

        legal_moves = generate_legal_moves(cur_board, cur_side)
        if not legal_moves:
            backup(search_path, -1.0)
            continue

        legal_actions = [move_to_action(m) for m in legal_moves]
        key = position_key(cur_board, cur_side)
        cached = cache.get(key)
        if cached is not None:
            logits, value = cached
            priors = normalize_logits(logits, legal_actions)
            node.children = {action: Node(prior) for action, prior in priors.items()}
            backup(search_path, float(value))
            continue

        pending.append((node, cur_board, cur_side, search_path, legal_actions, key))
        if len(pending) >= max(1, config.batch_size):
            flush_pending()

    flush_pending()
    return root


def select_action(root: Node, temperature: float) -> Optional[int]:
    if not root.children:
        return None
    actions = list(root.children.keys())
    visits = np.array([root.children[a].visit_count for a in actions], dtype=np.float32)

    if temperature <= 1e-4:
        return actions[int(np.argmax(visits))]

    visits = visits ** (1.0 / temperature)
    probs = visits / (visits.sum() + 1e-8)
    return int(np.random.choice(actions, p=probs))


def visit_counts(root: Node) -> Dict[int, int]:
    return {action: child.visit_count for action, child in root.children.items()}
