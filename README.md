# 自我进化中国象棋AI

这是一个纯前端界面 + 本地 Python 后端的中国象棋项目。棋盘展示、走子规则与判定在前端完成，**AI 走子由 Python 后端强制提供**（无前端 AI 回退）。

## 功能概览

- 网页对弈：人机 / AI 对 AI
- AI 搜索深度可调（前端参数映射到后端 MCTS 仿真数）
- 后端状态提示（离线时会显示“后端不可用”）
- Python 训练面板：可在网页中启动/停止训练并查看日志
- 规则判定：重复局面判和、长将判和
- 将军提示：状态栏显示“被将军”，并高亮将军子与被将军的帅

## 快速启动（推荐）

该项目需要 Python 后端提供 AI 走子能力。

```bash
python3 -m pip install -r backend/requirements.txt
python3 run.py
```

`run.py` 会同时启动：

- 后端：`http://127.0.0.1:8001`
- 前端：`http://127.0.0.1:8000`（端口会自动在 8000/8002/8003/8080 中选择可用项）

## 手动启动

1) 启动后端：

```bash
python3 -m backend.server
```

2) 启动前端静态服务：

```bash
python3 -m http.server 8000
```

浏览器访问：

```
http://localhost:8000
```

## Python 训练

页面右侧「Python 训练」卡片可直接启动/停止训练并查看进度与日志。

也可命令行训练：

```bash
python3 -m backend.train --iterations 200 --games-per-iter 8 --simulations 320
```

训练模型会保存至：

```
backend/checkpoints/latest.pt
```

## 使用说明

- 勾选 AI 控制红方/黑方即可启动 AI 走子
- 后端离线时会显示“后端不可用”，AI 将不会走子
- “AI 搜索深度”会影响后端 MCTS 仿真数
- 将军时状态栏会提示，并高亮将军子与被将军的帅

## 目录结构

- `index.html`：页面结构
- `styles.css`：棋盘与界面样式
- `app.js`：前端逻辑（规则、交互、与后端通信）
- `engine.js`：走子规则与判定
- `backend/`：Python 后端（MCTS + 训练）
- `run.py`：一键启动前后端
