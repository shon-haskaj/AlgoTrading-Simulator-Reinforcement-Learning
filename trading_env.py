from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None

    class _BaseEnv:
        pass

    class _Discrete:
        def __init__(self, n: int) -> None:
            self.n = n

        def sample(self) -> int:
            return int(np.random.randint(self.n))

    class _Box:
        def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: Any) -> None:
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Spaces:
        Discrete = _Discrete
        Box = _Box

    spaces = _Spaces()
else:
    _BaseEnv = gym.Env


@dataclass
class MarketSimConfig:
    episode_length: int = 512
    lookback: int = 32
    init_price: float = 100.0
    init_portfolio_value: float = 1.0
    max_position: float = 1.0
    trading_fee_bps: float = 1.0
    half_spread_bps: float = 1.0
    slippage_bps: float = 2.0
    impact_bps: float = 3.0
    risk_aversion: float = 0.02
    drawdown_limit: float = 0.35
    jump_prob: float = 0.01
    seed: int | None = None


class TradingMarketEnv(_BaseEnv):
    """
    A stylized but realistic market simulator for RL.

    Actions:
    - 0: target short position  -max_position
    - 1: target flat position    0
    - 2: target long position   +max_position

    Observation:
    [lookback log returns..., short_momentum, long_momentum, vol, drawdown, position]
    """

    metadata = {"render_modes": ["human", "notebook"]}

    def __init__(self, config: MarketSimConfig | None = None) -> None:
        self.config = config or MarketSimConfig()
        self.rng = np.random.default_rng(self.config.seed)

        obs_size = self.config.lookback + 5
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self._action_to_position = np.array(
            [-self.config.max_position, 0.0, self.config.max_position], dtype=np.float32
        )

        self.reset()

    def _generate_price_path(self) -> tuple[np.ndarray, np.ndarray]:
        total_steps = self.config.episode_length + self.config.lookback + 1
        prices = np.empty(total_steps, dtype=np.float64)
        returns = np.empty(total_steps - 1, dtype=np.float64)
        prices[0] = self.config.init_price

        # 0=bear, 1=sideways, 2=bull
        transition = np.array(
            [[0.90, 0.09, 0.01], [0.08, 0.84, 0.08], [0.02, 0.10, 0.88]], dtype=np.float64
        )
        drift = np.array([-0.0007, 0.0, 0.0008], dtype=np.float64)
        base_vol = np.array([0.018, 0.010, 0.013], dtype=np.float64)

        regime = 1
        vol = base_vol[regime]
        prev_ret = 0.0

        for t in range(total_steps - 1):
            regime = int(self.rng.choice(3, p=transition[regime]))
            target_vol = base_vol[regime]
            vol = 0.92 * vol + 0.08 * target_vol * (1.0 + 2.0 * abs(prev_ret))
            shock = self.rng.normal(0.0, vol)
            jump = 0.0
            if self.rng.random() < self.config.jump_prob:
                jump = self.rng.normal(0.0, 3.0 * vol)
            ret = drift[regime] + shock + jump
            returns[t] = ret
            prices[t + 1] = max(1e-6, prices[t] * np.exp(ret))
            prev_ret = ret
        return prices, returns

    def _obs(self) -> np.ndarray:
        start = self.t - self.config.lookback
        end = self.t
        log_rets = np.diff(np.log(self.prices[start : end + 1]))
        short_mom = float(np.mean(log_rets[-5:]))
        long_mom = float(np.mean(log_rets[-20:]))
        vol = float(np.std(log_rets[-20:]) + 1e-8)
        drawdown = float((self.peak_value - self.portfolio_value) / max(self.peak_value, 1e-8))
        obs = np.concatenate(
            [
                np.clip(log_rets, -0.2, 0.2),
                np.array([short_mom, long_mom, vol, drawdown, self.position], dtype=np.float64),
            ]
        )
        return obs.astype(np.float32)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.prices, self.raw_returns = self._generate_price_path()
        self.t = self.config.lookback
        self.end_t = self.config.lookback + self.config.episode_length - 1
        self.position = 0.0
        self.portfolio_value = self.config.init_portfolio_value
        self.peak_value = self.portfolio_value
        self.history: dict[str, list[float]] = {
            "t": [float(self.t)],
            "price": [float(self.prices[self.t])],
            "portfolio": [float(self.portfolio_value)],
            "position": [float(self.position)],
            "trade": [0.0],
            "reward": [0.0],
        }
        obs = self._obs()
        info = {"portfolio_value": self.portfolio_value, "position": self.position}
        return obs, info

    def step(self, action: int):
        action = int(action)
        target_pos = float(self._action_to_position[action])
        trade = target_pos - self.position

        fee = self.config.trading_fee_bps / 10000.0
        spread = self.config.half_spread_bps / 10000.0
        slippage = self.config.slippage_bps / 10000.0
        impact = self.config.impact_bps / 10000.0

        trade_cost_rate = fee + spread + slippage * abs(trade) + impact * (abs(trade) ** 2)
        trade_cost = self.portfolio_value * abs(trade) * trade_cost_rate

        price_t = self.prices[self.t]
        price_tp1 = self.prices[self.t + 1]
        asset_ret = (price_tp1 / price_t) - 1.0

        old_value = self.portfolio_value
        gross_pnl = old_value * target_pos * asset_ret
        self.portfolio_value = max(1e-8, old_value + gross_pnl - trade_cost)
        self.position = target_pos

        step_vol = abs(np.log(price_tp1 / price_t))
        reward = np.log(self.portfolio_value / old_value) - self.config.risk_aversion * (
            self.position**2
        ) * step_vol

        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = (self.peak_value - self.portfolio_value) / max(self.peak_value, 1e-8)

        self.t += 1
        terminated = self.t >= self.end_t
        truncated = drawdown >= self.config.drawdown_limit

        self.history["t"].append(float(self.t))
        self.history["price"].append(float(self.prices[self.t]))
        self.history["portfolio"].append(float(self.portfolio_value))
        self.history["position"].append(float(self.position))
        self.history["trade"].append(float(trade))
        self.history["reward"].append(float(reward))

        obs = self._obs()
        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "asset_return": asset_ret,
            "trade_cost": trade_cost,
            "drawdown": drawdown,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self, mode: str = "notebook"):
        import matplotlib.pyplot as plt

        if len(self.history["t"]) < 2:
            return None
        x = np.arange(len(self.history["t"]))
        prices = np.asarray(self.history["price"])
        pv = np.asarray(self.history["portfolio"])
        trades = np.asarray(self.history["trade"])

        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(x, prices, color="#1f77b4", label="Price")
        buy_idx = np.where(trades > 1e-8)[0]
        sell_idx = np.where(trades < -1e-8)[0]
        axes[0].scatter(buy_idx, prices[buy_idx], marker="^", color="#2ca02c", s=36, label="Buy")
        axes[0].scatter(sell_idx, prices[sell_idx], marker="v", color="#d62728", s=36, label="Sell")
        axes[0].set_ylabel("Price")
        axes[0].legend(loc="upper left")
        axes[0].grid(alpha=0.25)

        axes[1].plot(x, pv, color="#ff7f0e")
        axes[1].set_ylabel("Portfolio")
        axes[1].grid(alpha=0.25)

        axes[2].step(x, np.asarray(self.history["position"]), where="post", color="#9467bd")
        axes[2].set_ylabel("Position")
        axes[2].set_xlabel("Step")
        axes[2].set_yticks([-self.config.max_position, 0.0, self.config.max_position])
        axes[2].grid(alpha=0.25)

        fig.tight_layout()
        if mode == "human":
            plt.show()
        return fig


def rollout_episode(env: TradingMarketEnv, greedy_action_fn=None, max_steps: int | None = None) -> dict[str, Any]:
    obs, info = env.reset()
    rewards: list[float] = []
    max_steps = max_steps or env.config.episode_length

    for _ in range(max_steps):
        if greedy_action_fn is None:
            action = env.action_space.sample()
        else:
            action = int(greedy_action_fn(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break

    summary = {
        "steps": len(rewards),
        "reward_sum": float(np.sum(rewards)),
        "reward_mean": float(np.mean(rewards) if rewards else 0.0),
        "final_portfolio": float(info["portfolio_value"]),
        "max_portfolio": float(np.max(env.history["portfolio"])),
        "min_portfolio": float(np.min(env.history["portfolio"])),
    }
    return summary
