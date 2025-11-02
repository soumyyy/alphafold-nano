#!/usr/bin/env python3
"""Train a tiny actor-critic agent on CartPole-v1.

Saves:
- models/cartpole_policy.pt                (policy/value network parameters)
- logs/cartpole/training_log.npy           (array of [update, mean_return, policy_loss, value_loss])
- logs/cartpole/training_plot.png          (loss/return curves)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

@dataclass
class EpisodeBatch:
    states: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 64, n_actions: int = 2) -> None:
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.base(obs)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

def prepare_dirs() -> Dict[str, str]:
    paths = {
        "models": os.path.join("models"),
        "logs": os.path.join("logs", "cartpole"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths

def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    returns: List[float] = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.append(running)
    returns.reverse()
    return returns

def collect_batch(
    env: gym.Env,
    model: ActorCritic,
    episodes_per_update: int,
    gamma: float,
    device: torch.device,
) -> Tuple[EpisodeBatch, float]:
    all_states: List[torch.Tensor] = []
    all_actions: List[int] = []
    all_returns: List[float] = []
    all_advantages: List[float] = []
    returns_per_episode: List[float] = []

    model.eval()
    for _ in range(episodes_per_update):
        obs, _ = env.reset()
        done = False
        ep_states: List[np.ndarray] = []
        ep_actions: List[int] = []
        ep_rewards: List[float] = []
        ep_values: List[float] = []

        while not done:
            state_tensor = torch.from_numpy(obs).to(device=device, dtype=torch.float32)
            logits, value = model(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            ep_states.append(obs.astype(np.float32))
            ep_actions.append(action)
            ep_rewards.append(float(reward))
            ep_values.append(value.item())

            obs = next_obs

        ep_returns = compute_returns(ep_rewards, gamma)
        ep_advantages = [ret - val for ret, val in zip(ep_returns, ep_values)]

        all_states.extend(torch.from_numpy(np.stack(ep_states)))
        all_actions.extend(ep_actions)
        all_returns.extend(ep_returns)
        all_advantages.extend(ep_advantages)
        returns_per_episode.append(sum(ep_rewards))

    states_tensor = torch.stack(all_states).to(device=device, dtype=torch.float32)
    actions_tensor = torch.tensor(all_actions, dtype=torch.int64, device=device)
    returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=device)
    advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32, device=device)
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

    batch = EpisodeBatch(states_tensor, actions_tensor, returns_tensor, advantages_tensor)
    mean_return = float(np.mean(returns_per_episode)) if returns_per_episode else 0.0
    return batch, mean_return

def train(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    env = gym.make("CartPole-v1")
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model = ActorCritic(obs_dim, hidden_dim=args.hidden_dim, n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    paths = prepare_dirs()
    log_rows: List[Tuple[int, float, float, float]] = []

    for update in range(1, args.updates + 1):
        batch, mean_return = collect_batch(
            env,
            model,
            args.episodes_per_update,
            args.gamma,
            device,
        )

        model.train()
        optimizer.zero_grad()

        logits, values = model(batch.states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(batch.actions)
        entropy = dist.entropy().mean()

        policy_loss = -(log_probs * batch.advantages).mean()
        value_loss = nn.functional.mse_loss(values, batch.returns)
        loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        log_rows.append((update, mean_return, float(policy_loss.item()), float(value_loss.item())))

        if update % args.log_interval == 0:
            print(
                f"update={update:04d} mean_return={mean_return:.1f} "
                f"policy_loss={policy_loss.item():.3f} value_loss={value_loss.item():.3f}"
            )

    env.close()

    log_array = np.array(log_rows, dtype=np.float32)
    np.save(os.path.join(paths["logs"], "training_log.npy"), log_array)
    torch.save(model.state_dict(), os.path.join(paths["models"], "cartpole_policy.pt"))

    plot_training(log_array, os.path.join(paths["logs"], "training_plot.png"))
    print("Training complete. Artifacts saved under models/ and logs/cartpole/.")

def plot_training(log_array: np.ndarray, out_path: str) -> None:
    updates = log_array[:, 0]
    mean_returns = log_array[:, 1]
    policy_loss = log_array[:, 2]
    value_loss = log_array[:, 3]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(updates, mean_returns, label="Mean Return")
    axes[0].axhline(475, color="red", linestyle="--", label="Target 475")
    axes[0].set_ylabel("Mean Return")
    axes[0].legend()

    axes[1].plot(updates, policy_loss, label="Policy Loss")
    axes[1].plot(updates, value_loss, label="Value Loss")
    axes[1].set_xlabel("Update")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CartPole actor-critic trainer")
    parser.add_argument("--updates", type=int, default=500, help="Number of policy updates")
    parser.add_argument("--episodes-per-update", type=int, default=10, help="Episodes collected before each update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy regularization coefficient")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss weight")
    parser.add_argument("--grad-clip", type=float, default=0.5, help="Gradient clip norm")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer width")
    parser.add_argument("--log-interval", type=int, default=20, help="Number of updates between console logs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
