# Python 神经网络后端

此目录提供 Python 版神经网络 + MCTS 的本地后端，供网页 UI 调用。

## 安装

- 安装依赖: `python3 -m pip install -r backend/requirements.txt`

## 启动后端

- `python3 -m backend.server`

默认监听 `http://127.0.0.1:8001`。

## 训练

- `python3 -m backend.train --iterations 200 --games-per-iter 8 --simulations 320`

训练完成的模型保存在 `backend/checkpoints/latest.pt`。
