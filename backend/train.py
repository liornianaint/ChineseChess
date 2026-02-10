from __future__ import annotations

import argparse
import re
import random
import threading
import shutil
import sys
import importlib
from contextlib import nullcontext
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


def maybe_warn_cuda_install(device: torch.device) -> None:
    if device.type != "cpu":
        return
    if torch.backends.mps.is_available():
        return
    if shutil.which("nvidia-smi") or Path("/proc/driver/nvidia/version").exists():
        print(
            "检测到 NVIDIA GPU，但当前 PyTorch 未启用 CUDA。"
            "可执行：pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio",
            flush=True,
        )


def get_autocast(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return torch.cuda.amp.autocast(dtype=torch.float16)


def get_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def save_checkpoint(
    path: Path,
    model: XiangqiNet,
    optimizer: torch.optim.Optimizer,
    step: int,
    total_games: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "total_games": int(total_games),
        },
        path,
    )


def load_checkpoint(path: Path, model: XiangqiNet, optimizer: torch.optim.Optimizer) -> Tuple[int, int]:
    if not path.exists():
        return 0, 0
    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data.get("model", {}))
    optimizer.load_state_dict(data.get("optimizer", {}))
    return int(data.get("step", 0)), int(data.get("total_games", 0))


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


def train_step(
    model: XiangqiNet,
    optimizer: torch.optim.Optimizer,
    batch: List[TrainingExample],
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = False,
) -> float:
    if not batch:
        return 0.0
    model.train()
    boards, sides, policies, values = zip(*batch)
    inputs = torch.stack([encode_board(b, s) for b, s in zip(boards, sides)]).to(device)
    target_policy = torch.from_numpy(np.stack(policies)).to(device=device, dtype=torch.float32)
    target_value = torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(1)

    optimizer.zero_grad(set_to_none=True)
    autocast_ctx = get_autocast(device, use_amp)
    with autocast_ctx:
        pred_policy, pred_value = model(inputs)
        log_probs = F.log_softmax(pred_policy, dim=1)
        policy_loss = -(target_policy * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(pred_value, target_value)
        loss = policy_loss + value_loss

    if use_amp and scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
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
    parser.add_argument("--mcts-batch-size", type=int, default=256)
    parser.add_argument("--batches-per-epoch", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=str, default=default_checkpoint)
    args = parser.parse_args()

    device = get_device()
    maybe_warn_cuda_install(device)
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    model = XiangqiNet().to(device)

    if device.type == "cuda":
        auto_workers = min(args.games_per_iter, max(4, args.mcts_batch_size // 32))
        args.selfplay_workers = max(args.selfplay_workers, auto_workers)

        if hasattr(torch, "compile"):
            try:
                triton_ok = True
                try:
                    import triton  # noqa: F401
                except Exception:
                    triton_ok = False
                if sys.platform.startswith("win") and not triton_ok:
                    raise RuntimeError("triton not available on this setup")
                model = torch.compile(model)
            except Exception:
                try:
                    torch_dynamo = importlib.import_module("torch._dynamo")
                    torch_dynamo.config.suppress_errors = True
                except Exception:
                    pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = get_grad_scaler(use_amp)

    checkpoint_dir = Path(args.checkpoint_dir)
    step, total_games = load_checkpoint(checkpoint_dir / "latest.pt", model, optimizer)

    buffer = ReplayBuffer(args.buffer_size)
    games_done_total = total_games

    for iteration in range(args.iterations):
        selfplay = SelfPlayConfig(
            simulations=args.simulations,
            max_moves=args.max_moves,
            temperature_moves=args.temperature_moves,
            mcts_batch_size=args.mcts_batch_size,
        )
        total_moves = 0
        train_unit_weight = max(1, args.max_moves // 5)
        train_units_total = args.epochs * args.batches_per_epoch * train_unit_weight
        selfplay_moves_done = 0
        training_units_done = 0
        games_done = 0
        moves_done_finished = 0
        last_pct = -1
        iter_label = f"{iteration + 1}/{args.iterations}"
        progress_lock = threading.Lock()

        def estimate_total_units() -> float:
            if games_done > 0:
                avg_moves = moves_done_finished / games_done
            else:
                avg_moves = args.max_moves
            expected_selfplay = max(selfplay_moves_done, avg_moves * args.games_per_iter)
            return expected_selfplay + train_units_total

        def emit_progress(phase: str) -> None:
            nonlocal last_pct
            total_units = estimate_total_units()
            done_units = selfplay_moves_done + training_units_done
            pct = 0
            if total_units > 0:
                pct = int((done_units / total_units) * 100)
            if pct == last_pct:
                return
            last_pct = pct
            print(f"progress iter {iter_label} pct {pct} phase {phase}", flush=True)

        def advance_selfplay(units: int) -> None:
            nonlocal selfplay_moves_done
            if units <= 0:
                return
            with progress_lock:
                selfplay_moves_done += units
                emit_progress("selfplay")

        def finish_game(moves: int) -> None:
            nonlocal games_done, moves_done_finished
            with progress_lock:
                games_done += 1
                moves_done_finished += moves
                emit_progress("selfplay")

        def advance_train(units: int) -> None:
            nonlocal training_units_done
            if units <= 0:
                return
            with progress_lock:
                training_units_done += units
                emit_progress("train")
        workers = max(1, min(args.selfplay_workers, args.games_per_iter))
        if workers <= 1:
            for _ in range(args.games_per_iter):
                examples, moves, _result = self_play_game(model, selfplay, device, progress_cb=advance_selfplay)
                buffer.extend(examples)
                total_moves += moves
                finish_game(moves)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        self_play_game,
                        model,
                        selfplay,
                        device,
                        advance_selfplay,
                    )
                    for _ in range(args.games_per_iter)
                ]
                for future in as_completed(futures):
                    examples, moves, _result = future.result()
                    buffer.extend(examples)
                    total_moves += moves
                    finish_game(moves)

        games_done_total += games_done

        if len(buffer) == 0:
            continue

        for _ in range(args.epochs):
            for _ in range(args.batches_per_epoch):
                batch = buffer.sample(args.batch_size)
                train_step(model, optimizer, batch, device, scaler=scaler, use_amp=use_amp)
                advance_train(train_unit_weight)

        step += 1
        save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, step, games_done_total)
        if step % 5 == 0:
            save_checkpoint(checkpoint_dir / f"model_{step}.pt", model, optimizer, step, games_done_total)
        prune_checkpoints(checkpoint_dir, keep=3)
        print(f"iter {iteration + 1}/{args.iterations}: buffer={len(buffer)} moves={total_moves}", flush=True)


if __name__ == "__main__":
    main()
