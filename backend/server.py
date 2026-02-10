from __future__ import annotations

from collections import deque
from pathlib import Path
import time
import subprocess
import sys
import threading
from typing import Deque, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .engine import Board, generate_legal_moves, position_key
from .encoding import action_to_move, move_to_action
from .model import XiangqiNet
from .mcts import MCTSConfig, run_mcts, select_action, visit_counts
from .opening_book import opening_book_move


BASE_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = BASE_DIR / "backend" / "checkpoints" / "latest.pt"
TRAIN_LOG_LIMIT = 5000

MODEL_LOCK = threading.Lock()
MODEL_INFO: Dict[str, Optional[object]] = {
    "loaded": False,
    "step": None,
    "mtime": None,
    "path": str(CHECKPOINT_PATH),
    "total_games": None,
}


class MoveRequest(BaseModel):
    board: List[Optional[str]] = Field(..., min_length=90, max_length=90)
    side: str
    sims: int = 320
    temperature: float = 0.0
    timeLimitMs: int = 0
    useBook: bool = True


class MoveResponse(BaseModel):
    move: Optional[dict]
    stats: dict


class TrainStartRequest(BaseModel):
    iterations: int = 600
    games_per_iter: int = 12
    selfplay_workers: int = 4
    simulations: int = 480
    max_moves: int = 220
    temperature_moves: int = 12
    batch_size: int = 128
    epochs: int = 2
    buffer_size: int = 15000
    lr: float = 2e-3


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def enable_cuda_performance(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def load_model(device: torch.device) -> XiangqiNet:
    global MODEL_INFO
    model = XiangqiNet().to(device)
    if CHECKPOINT_PATH.exists():
        data = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(data.get("model", data))
        step = int(data.get("step", 0)) if isinstance(data, dict) else 0
        mtime = CHECKPOINT_PATH.stat().st_mtime
        total_games = int(data.get("total_games", 0)) if isinstance(data, dict) else 0
        MODEL_INFO = {
            "loaded": True,
            "step": step,
            "mtime": mtime,
            "path": str(CHECKPOINT_PATH),
            "total_games": total_games,
        }
    else:
        MODEL_INFO = {
            "loaded": False,
            "step": None,
            "mtime": None,
            "path": str(CHECKPOINT_PATH),
            "total_games": None,
        }
    model.eval()
    return model


def reload_model() -> bool:
    global MODEL
    if not CHECKPOINT_PATH.exists():
        MODEL_INFO["loaded"] = False
        MODEL_INFO["step"] = None
        MODEL_INFO["mtime"] = None
        MODEL_INFO["total_games"] = None
        return False
    with MODEL_LOCK:
        MODEL = load_model(DEVICE)
    return True


def validate_board(board: List[Optional[str]]) -> Board:
    if len(board) != 90:
        raise ValueError("board must have length 90")
    allowed = set(["K", "A", "B", "N", "R", "C", "P", "k", "a", "b", "n", "r", "c", "p"])
    normalized: Board = []
    for piece in board:
        if piece is None:
            normalized.append(None)
            continue
        if piece not in allowed:
            raise ValueError(f"invalid piece: {piece}")
        normalized.append(piece)
    return normalized


class TrainingManager:
    def __init__(self) -> None:
        self.process: Optional[subprocess.Popen] = None
        self.reader: Optional[threading.Thread] = None
        self.lines: Deque[str] = deque(maxlen=TRAIN_LOG_LIMIT)
        self.running = False
        self.last_exit_code: Optional[int] = None
        self.config: Dict[str, float] = {}
        self.error: Optional[str] = None
        self.start_time: Optional[float] = None

    def start(self, config: TrainStartRequest) -> bool:
        if self.process and self.process.poll() is None:
            return False

        cmd = [
            sys.executable,
            "-m",
            "backend.train",
            "--iterations",
            str(config.iterations),
            "--games-per-iter",
            str(config.games_per_iter),
            "--selfplay-workers",
            str(config.selfplay_workers),
            "--simulations",
            str(config.simulations),
            "--max-moves",
            str(config.max_moves),
            "--temperature-moves",
            str(config.temperature_moves),
            "--batch-size",
            str(config.batch_size),
            "--epochs",
            str(config.epochs),
            "--buffer-size",
            str(config.buffer_size),
            "--lr",
            str(config.lr),
        ]

        self.process = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.lines.clear()
        self.config = config.dict()
        self.error = None
        self.running = True
        self.start_time = time.time()
        self.lines.append("训练已启动，正在自我对弈生成数据…")
        self.last_exit_code = None
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()
        return True

    def _read_output(self) -> None:
        assert self.process is not None
        stdout = self.process.stdout
        try:
            if stdout:
                for line in stdout:
                    self.lines.append(line.rstrip())
        except Exception as exc:  # pragma: no cover - best effort
            self.error = str(exc)
        finally:
            code = self.process.wait()
            self.last_exit_code = code
            self.running = False
            self.start_time = None
            if code == 0:
                if reload_model():
                    self.lines.append("训练完成，模型已自动加载。")
                else:
                    self.lines.append("训练完成，但未找到模型文件。")
            else:
                self.lines.append(f"训练退出 code={code}")

    def stop(self) -> bool:
        if not self.process or self.process.poll() is not None:
            return False
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
        if reload_model():
            self.lines.append("训练已停止，模型已自动加载。")
        else:
            self.lines.append("训练已停止，但未找到模型文件。")
        return True

    def status(self) -> Dict[str, object]:
        return {
            "running": self.running,
            "exitCode": self.last_exit_code,
            "lines": list(self.lines),
            "config": self.config,
            "error": self.error,
            "pid": getattr(self.process, "pid", None),
            "startTime": self.start_time,
        }


app = FastAPI(title="Xiangqi Neural Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = get_device()
enable_cuda_performance(DEVICE)
MODEL = load_model(DEVICE)
TRAINER = TrainingManager()


@app.on_event("startup")
def _startup_load_model() -> None:
    reload_model()


@app.get("/health")
def health() -> dict:
    if not CHECKPOINT_PATH.exists():
        MODEL_INFO["loaded"] = False
        MODEL_INFO["step"] = None
        MODEL_INFO["mtime"] = None
        MODEL_INFO["total_games"] = None
    return {"status": "ok", "device": str(DEVICE), "model": MODEL_INFO}


@app.get("/")
def root() -> dict:
    return {"status": "ok", "message": "Xiangqi backend running", "device": str(DEVICE)}


@app.post("/ai/move", response_model=MoveResponse)
def ai_move(req: MoveRequest):
    if not CHECKPOINT_PATH.exists() or not MODEL_INFO.get("loaded"):
        raise HTTPException(status_code=409, detail="model_not_loaded")
    try:
        board = validate_board(req.board)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    side = req.side
    if side not in ("r", "b"):
        raise HTTPException(status_code=400, detail="side must be 'r' or 'b'")

    if req.useBook:
        book_move = opening_book_move(board, side)
        if book_move:
            return {
                "move": {"from": book_move.from_idx, "to": book_move.to_idx},
                "stats": {"sims": 0, "visits": 0, "book": True, "boardKey": position_key(board, side)},
            }

    sim_count = max(16, min(req.sims, 4000))
    if DEVICE.type == "cuda":
        mcts_batch = min(256, sim_count)
    else:
        mcts_batch = min(64, sim_count)
    config = MCTSConfig(simulations=sim_count, batch_size=mcts_batch)
    with MODEL_LOCK:
        root = run_mcts(MODEL, board, side, config, DEVICE)
        action = select_action(root, req.temperature)
        visits = visit_counts(root)

    legal_moves = generate_legal_moves(board, side)
    legal_actions = {move_to_action(m) for m in legal_moves}
    used_fallback = False

    if action is None or action not in legal_actions:
        if legal_actions:
            if visits:
                best_action = max(legal_actions, key=lambda a: visits.get(a, 0))
                if visits.get(best_action, 0) == 0:
                    best_action = next(iter(legal_actions))
            else:
                best_action = next(iter(legal_actions))
            action = best_action
            used_fallback = True
        else:
            return {
                "move": None,
                "stats": {"sims": config.simulations, "visits": 0, "book": False, "boardKey": position_key(board, side)},
            }

    move = action_to_move(action)
    return {
        "move": {"from": move.from_idx, "to": move.to_idx},
        "stats": {
            "sims": config.simulations,
            "visits": visits.get(action, 0),
            "book": False,
            "fallback": used_fallback,
            "boardKey": position_key(board, side),
        },
    }


@app.post("/train/start")
def train_start(req: TrainStartRequest):
    if not TRAINER.start(req):
        raise HTTPException(status_code=409, detail="training already running")
    return {"status": "started"}


@app.post("/train/stop")
def train_stop():
    if TRAINER.stop():
        return {"status": "stopping"}
    return {"status": "idle"}


@app.get("/train/status")
def train_status():
    return TRAINER.status()


@app.post("/model/reload")
def model_reload():
    if reload_model():
        return {"status": "reloaded"}
    return {"status": "missing"}


def main() -> None:
    import uvicorn

    uvicorn.run("backend.server:app", host="127.0.0.1", port=8001, reload=False)


if __name__ == "__main__":
    main()
