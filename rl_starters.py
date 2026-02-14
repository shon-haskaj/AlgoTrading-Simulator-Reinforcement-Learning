from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ToyTradingMDP:
    """
    Small model-based MDP for DP methods (policy/value iteration).
    States: 0=bear, 1=sideways, 2=bull
    Actions: 0=short, 1=flat, 2=long
    """

    gamma: float = 0.95

    def __post_init__(self) -> None:
        self.n_states = 3
        self.n_actions = 3
        self.action_pos = np.array([-1.0, 0.0, 1.0], dtype=np.float64)

        self.P = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=np.float64)
        self.R = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

        base_transition = np.array(
            [[0.78, 0.20, 0.02], [0.18, 0.64, 0.18], [0.03, 0.24, 0.73]], dtype=np.float64
        )
        state_return = np.array([-0.003, 0.0, 0.003], dtype=np.float64)
        risk_penalty = 0.0005

        for s in range(self.n_states):
            for a in range(self.n_actions):
                self.P[s, a] = base_transition[s]
                pos = self.action_pos[a]
                self.R[s, a] = pos * state_return[s] - risk_penalty * (pos**2)

    def sample_step(self, state: int, action: int, rng: np.random.Generator) -> tuple[int, float]:
        next_state = int(rng.choice(self.n_states, p=self.P[state, action]))
        reward = float(self.R[state, action])
        return next_state, reward


def policy_iteration(mdp: ToyTradingMDP, tol: float = 1e-10, max_iter: int = 500):
    policy = np.zeros(mdp.n_states, dtype=np.int64)
    V = np.zeros(mdp.n_states, dtype=np.float64)

    for _ in range(max_iter):
        for _ in range(max_iter):
            prev = V.copy()
            for s in range(mdp.n_states):
                a = policy[s]
                V[s] = mdp.R[s, a] + mdp.gamma * np.dot(mdp.P[s, a], prev)
            if np.max(np.abs(V - prev)) < tol:
                break

        stable = True
        for s in range(mdp.n_states):
            q_vals = np.empty(mdp.n_actions, dtype=np.float64)
            for a in range(mdp.n_actions):
                q_vals[a] = mdp.R[s, a] + mdp.gamma * np.dot(mdp.P[s, a], V)
            best = int(np.argmax(q_vals))
            if best != policy[s]:
                stable = False
            policy[s] = best
        if stable:
            break

    return policy, V


def td0_policy_evaluation(
    mdp: ToyTradingMDP,
    policy: np.ndarray,
    episodes: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 0,
):
    rng = np.random.default_rng(rng_seed)
    V = np.zeros(mdp.n_states, dtype=np.float64)

    for _ in range(episodes):
        s = int(rng.integers(0, mdp.n_states))
        for _ in range(50):
            a = int(policy[s])
            ns, r = mdp.sample_step(s, a, rng)
            td_target = r + mdp.gamma * V[ns]
            V[s] += alpha * (td_target - V[s])
            s = ns
    return V


def mc_policy_evaluation(
    mdp: ToyTradingMDP,
    policy: np.ndarray,
    episodes: int = 2000,
    rng_seed: int = 0,
):
    rng = np.random.default_rng(rng_seed)
    V = np.zeros(mdp.n_states, dtype=np.float64)
    returns_sum = np.zeros(mdp.n_states, dtype=np.float64)
    returns_count = np.zeros(mdp.n_states, dtype=np.float64)

    for _ in range(episodes):
        s = int(rng.integers(0, mdp.n_states))
        traj_s: list[int] = []
        traj_r: list[float] = []
        for _ in range(50):
            a = int(policy[s])
            ns, r = mdp.sample_step(s, a, rng)
            traj_s.append(s)
            traj_r.append(r)
            s = ns

        G = 0.0
        seen: set[int] = set()
        for t in range(len(traj_s) - 1, -1, -1):
            G = mdp.gamma * G + traj_r[t]
            st = traj_s[t]
            if st in seen:
                continue
            seen.add(st)
            returns_sum[st] += G
            returns_count[st] += 1.0
            V[st] = returns_sum[st] / max(returns_count[st], 1.0)
    return V


class UniformDiscretizer:
    """
    Converts continuous observations into discrete tuples for tabular methods.
    """

    def __init__(self, low: np.ndarray, high: np.ndarray, bins: int = 7):
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.bins = int(bins)
        self.edges = [
            np.linspace(self.low[i], self.high[i], self.bins + 1)[1:-1] for i in range(len(self.low))
        ]

    def transform(self, obs: np.ndarray) -> tuple[int, ...]:
        obs = np.asarray(obs, dtype=np.float64)
        ids = [int(np.digitize(obs[i], self.edges[i])) for i in range(obs.shape[0])]
        return tuple(ids)


def q_learning_tabular(
    env,
    discretizer: UniformDiscretizer,
    episodes: int = 500,
    alpha: float = 0.08,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    rng_seed: int = 0,
):
    rng = np.random.default_rng(rng_seed)
    q_table: dict[tuple[int, ...], np.ndarray] = {}

    def q_row(state: tuple[int, ...]) -> np.ndarray:
        if state not in q_table:
            q_table[state] = np.zeros(env.action_space.n, dtype=np.float64)
        return q_table[state]

    episode_rewards = []
    for ep in range(episodes):
        eps = epsilon_end + (epsilon_start - epsilon_end) * max(0.0, 1.0 - ep / episodes)
        obs, _ = env.reset()
        s = discretizer.transform(obs)
        total_r = 0.0

        while True:
            if rng.random() < eps:
                a = int(rng.integers(0, env.action_space.n))
            else:
                a = int(np.argmax(q_row(s)))
            next_obs, r, term, trunc, _ = env.step(a)
            ns = discretizer.transform(next_obs)

            td_target = r + gamma * np.max(q_row(ns)) * (0.0 if (term or trunc) else 1.0)
            q_row(s)[a] += alpha * (td_target - q_row(s)[a])

            s = ns
            total_r += r
            if term or trunc:
                break
        episode_rewards.append(total_r)
    return q_table, np.asarray(episode_rewards, dtype=np.float64)

