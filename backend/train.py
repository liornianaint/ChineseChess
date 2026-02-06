from __future__ import annotations

import argparse
import re
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .encoding import encode_board
from .model import XiangqiNet
from .selfplay import SelfPlayConfig, TrainingExample, self_play_game


class ReplayBuffer:
    def __init__(self, max_size: int):
        self.buffer: Deque[TrainingExample] = deque(maxlen=max_size)

    def extend(self, items: List[TrainingExample]) -> None:
        self.buffer.extend(items)

    def sample(self, batch_size: int) -> List[TrainingExample]:
        return random.sample(self.buffer, k=min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(path: Path, model: XiangqiNet, optimizer: torch.optim.Optimizer, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        path,
    )


def load_checkpoint(path: Path, model: XiangqiNet, optimizer: torch.optim.Optimizer) -> int:
    if not path.exists():
        return 0
    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data.get("model", {}))
    optimizer.load_state_dict(data.get("optimizer", {}))
    return int(data.get("step", 0))


def prune_checkpoints(checkpoint_dir: Path, keep: int = 3) -> None:
    if keep < 1:
        return
    model_paths = []
    for path in checkpoint_dir.glob("model_*.pt"):
        match = re.match(r"model_(\d+)\.pt$", path.name)
        if not match:
            continue
        model_paths.append((int(match.group(1)), path))

    model_paths.sort(key=lambda item: item[0], reverse=True)
    for _step, path in model_paths[max(0, keep - 1) :]:
        if path.exists():
            path.unlink()


def train_step(model: XiangqiNet, optimizer: torch.optim.Optimizer, batch: List[TrainingExample], device: torch.device) -> float:
    if not batch:
        return 0.0
    model.train()
    boards, sides, policies, values = zip(*batch)
    inputs = torch.stack([encode_board(b, s) for b, s in zip(boards, sides)]).to(device)
    target_policy = torch.from_numpy(np.stack(policies)).to(device=device, dtype=torch.float32)
    target_value = torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(1)

    optimizer.zero_grad(set_to_none=True)
    pred_policy, pred_value = model(inputs)

    log_probs = F.log_softmax(pred_policy, dim=1)
    policy_loss = -(target_policy * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(pred_value, target_value)
    loss = policy_loss + value_loss

    loss.backward()
    optimizer.step()
    return float(loss.item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Xiangqi self-play training")
    default_checkpoint = str(Path(__file__).resolve().parents[1] / "backend" / "checkpoints")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--games-per-iter", type=int, default=8)
    parser.add_argument("--selfplay-workers", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=240)
    parser.add_argument("--max-moves", type=int, default=220)
    parser.add_argument("--temperature-moves", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--buffer-size", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--mcts-batch-size", type=int, default=128)
    parser.add_argument("--batches-per-epoch", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=str, default=default_checkpoint)
    args = parser.parse_args()

    device = get_device()
    model = XiangqiNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    checkpoint_dir = Path(args.checkpoint_dir)
    step = load_checkpoint(checkpoint_dir / "latest.pt", model, optimizer)

    buffer = ReplayBuffer(args.buffer_size)

    for iteration in range(args.iterations):
        selfplay = SelfPlayConfig(
            simulations=args.simulations,
            max_moves=args.max_moves,
            temperature_moves=args.temperature_moves,
            mcts_batch_size=args.mcts_batch_size,
        )
        total_moves = 0
        train_unit_weight = max(1, args.max_moves // 5)
        total_units = (args.games_per_iter * args.max_moves) + (args.epochs * args.batches_per_epoch * train_unit_weight)
        done_units = 0
        last_pct = -1
        iter_label = f"{iteration + 1}/{args.iterations}"
        progress_lock = threading.Lock()

        def emit_progress(units: int, phase: str) -> None:
            nonlocal done_units, last_pct
            if units <= 0:
                return
            with progress_lock:
                done_units += units
                pct = 0
                if total_units > 0:
                    pct = int((done_units / total_units) * 100)
                if pct == last_pct:
                    return
                last_pct = pct
                print(f"progress iter {iter_label} pct {pct} phase {phase}", flush=True)
        workers = max(1, min(args.selfplay_workers, args.games_per_iter))
        if workers <= 1:
            for _ in range(args.games_per_iter):
                examples, moves, _result = self_play_game(model, selfplay, device, progress_cb=lambda n: emit_progress(n, "selfplay"))
                buffer.extend(examples)
                total_moves += moves
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        self_play_game,
                        model,
                        selfplay,
                        device,
                        lambda n: emit_progress(n, "selfplay"),
                    )
                    for _ in range(args.games_per_iter)
                ]
                for future in as_completed(futures):
                    examples, moves, _result = future.result()
                    buffer.extend(examples)
                    total_moves += moves

        if len(buffer) == 0:
            continue

        for _ in range(args.epochs):
            for _ in range(args.batches_per_epoch):
                batch = buffer.sample(args.batch_size)
                train_step(model, optimizer, batch, device)
                emit_progress(train_unit_weight, "train")

        if done_units < total_units:
            emit_progress(total_units - done_units, "save")

        step += 1
        save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, step)
        if step % 5 == 0:
            save_checkpoint(checkpoint_dir / f"model_{step}.pt", model, optimizer, step)
        prune_checkpoints(checkpoint_dir, keep=3)
        print(f"iter {iteration + 1}/{args.iterations}: buffer={len(buffer)} moves={total_moves}", flush=True)


if __name__ == "__main__":
    main()
