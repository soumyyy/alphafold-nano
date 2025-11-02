#!/usr/bin/env python3
"""Tabular Q-learning for FrozenLake-v1 (8x8, slippery=True).

Saves:
- models/frozenlake_q.npy                  (Q-table)
- logs/frozenlake/win_rate.npy             (win rate per eval window)
- logs/frozenlake/training_plot.png        (win-rate curve)
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

def prepare_dirs() -> Dict[str, str]:
    paths = {
        "models": os.path.join("models"),
        "logs": os.path.join("logs", "frozenlake"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths

def linear_schedule(step: int, max_steps: int, start: float, end: float) -> float:
    fraction = min(step / max_steps, 1.0)
    return start + fraction * (end - start)

def run_episode(env: gym.Env, Q: np.ndarray, epsilon: float, gamma: float, alpha: float, rng: np.random.Generator) -> Tuple[float, bool]:
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    success = False

    while not done:
        if rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q[state]))

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        best_next = np.max(Q[next_state]) if not done else 0.0
        td_target = reward + gamma * best_next
        td_error = td_target - Q[state, action]
        Q[state, action] += alpha * td_error

        state = next_state
        total_reward += reward
        if terminated and reward > 0:
            success = True

    return total_reward, success

def train(args: argparse.Namespace) -> None:
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    paths = prepare_dirs()
    rng = np.random.default_rng(args.seed)

    success_window: List[bool] = []
    win_rates: List[Tuple[int, float]] = []

    for episode in range(1, args.episodes + 1):
        epsilon = linear_schedule(episode, args.epsilon_decay_episodes, args.epsilon_start, args.epsilon_end)
        alpha = linear_schedule(episode, args.alpha_decay_episodes, args.alpha_start, args.alpha_end)

        _, success = run_episode(env, Q, epsilon, args.gamma, alpha, rng)
        success_window.append(success)
        if len(success_window) > args.eval_window:
            success_window.pop(0)

        if episode % args.log_interval == 0:
            window_rate = float(np.mean(success_window)) if success_window else 0.0
            win_rates.append((episode, window_rate))
            print(f"episode={episode:05d} epsilon={epsilon:.3f} alpha={alpha:.3f} win_rate={window_rate:.2f}")

    env.close()

    win_array = np.array(win_rates, dtype=np.float32)
    np.save(os.path.join(paths["logs"], "win_rate.npy"), win_array)
    np.save(os.path.join(paths["models"], "frozenlake_q.npy"), Q)

    plot_training(win_array, os.path.join(paths["logs"], "training_plot.png"))
    print("Training complete. Artifacts saved under models/ and logs/frozenlake/.")

def plot_training(win_array: np.ndarray, out_path: str) -> None:
    if win_array.size == 0:
        print("No win-rate data to plot.")
        return

    episodes = win_array[:, 0]
    win_rates = win_array[:, 1]

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, win_rates, label="Win Rate (per window)")
    plt.axhline(0.8, color="red", linestyle="--", label="Target 0.80")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FrozenLake tabular Q-learning trainer")
    parser.add_argument("--episodes", type=int, default=60000, help="Training episodes")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--epsilon-decay-episodes", type=int, default=60000, help="Episodes over which epsilon decays")
    parser.add_argument("--alpha-start", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--alpha-end", type=float, default=0.01, help="Final learning rate")
    parser.add_argument("--alpha-decay-episodes", type=int, default=60000, help="Episodes over which alpha decays")
    parser.add_argument("--eval-window", type=int, default=100, help="Window size for win-rate calculation")
    parser.add_argument("--log-interval", type=int, default=200, help="Episodes between logging/win-rate capture")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    np.random.seed(args.seed)
    return args

def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
