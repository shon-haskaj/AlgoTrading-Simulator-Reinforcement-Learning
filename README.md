# RL Trading Environment Guide

## Files
- `trading_env.py`: realistic market simulator (Gym-style API: `reset`, `step`, `render`)
- `rl_starters.py`: starter RL methods for learning progression
- `Untitled.ipynb`: ready-to-run notebook in JupyterLab

## Environment Interface

### Action space
- `Discrete(3)`
- `0 = short`, `1 = flat`, `2 = long`

### State (observation) space
- `Box(shape=(lookback + 5,))`
- Features:
- last `lookback` log returns
- short momentum (last 5 returns mean)
- long momentum (last 20 returns mean)
- rolling volatility
- current drawdown
- current position

### Reward
- `log(portfolio_t+1 / portfolio_t) - risk_penalty`
- Includes:
- PnL from position and price move
- trading costs (fee, spread, slippage, impact)
- risk penalty on position during volatile steps

## Model/Method Summary
- Policy iteration / value methods:
- use `ToyTradingMDP` in `rl_starters.py` first (small, model-based, easy to reason about)
- TD and Monte Carlo evaluation:
- use `td0_policy_evaluation` and `mc_policy_evaluation` on the toy MDP
- Q-learning:
- use `q_learning_tabular` with `UniformDiscretizer` on the realistic env
- DQN later.

## Run in Notebook
1. Open `Untitled.ipynb` in JupyterLab.
2. Run cells top to bottom.
3. Tune `MarketSimConfig` to change realism/difficulty (`spread`, `slippage`, `impact`, `drawdown_limit`, etc.).

## Dependencies
- `numpy`
- `matplotlib`
- optional: `gymnasium` (recommended for plug-and-play with RL libraries)
