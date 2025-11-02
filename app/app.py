#!/usr/bin/env python3
"""Streamlit dashboard for alphafold-nano.

Tabs:
- CartPole: training curves, quick greedy evaluation.
- FrozenLake: win-rate curve, evaluation.
- AlphaFold-nano: overlay toy backbone with AlphaFold CA trace and deviation metric.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn
from torch.distributions import Categorical

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

st.set_page_config(page_title="AlphaFold-nano Dashboard", layout="wide")
plt.style.use("dark_background")

st.markdown(
    """
    <style>
    :root {
        --bg-color: #050505;
        --panel-color: #111111;
        --accent-color: #4CC9F0;
        --accent-secondary: #F72585;
        --text-color: #F5F5F5;
        --muted-color: #C3C3C3;
    }
    body, .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
    }
    header {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .floating-nav {
        position: sticky;
        top: 1rem;
        z-index: 999;
        margin-bottom: 2.5rem;
    }
    .floating-nav div[data-testid="column"] {
        display: flex;
        justify-content: center;
    }
    .floating-nav div[data-testid="stButton"] > button {
        width: 110px;
        height: 110px;
        border-radius: 28px;
        background: rgba(20, 20, 20, 0.92);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0px 20px 40px rgba(0, 0, 0, 0.45);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.4rem;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        cursor: pointer;
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--muted-color);
        text-align: center;
        background-image: none;
        white-space: pre-line;
    }
    .floating-nav div[data-testid="stButton"] > button::first-line {
        font-size: 2rem;
    }
    .floating-nav div[data-testid="stButton"] > button:hover {
        transform: translateY(-6px) scale(1.05);
        box-shadow: 0px 24px 50px rgba(67, 97, 238, 0.35);
        border-color: rgba(76, 201, 240, 0.5);
    }
    .floating-nav div[data-testid="stButton"] > button span {
        display: block;
        line-height: 1.2;
    }
    .stMetric {
        background: rgba(255,255,255,0.04);
        border-radius: 16px;
        padding: 0.75rem 1rem;
    }
    .stMetric label, .stMetric span, .stMetric div {
        color: var(--text-color) !important;
    }
    .stExpander {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


@st.cache_resource(show_spinner=False)
def load_cartpole_model() -> ActorCritic | None:
    model_path = MODELS_DIR / "cartpole_policy.pt"
    if not model_path.exists():
        return None
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()
    model = ActorCritic(obs_dim, n_actions=n_actions)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_data(show_spinner=False)
def load_cartpole_log() -> pd.DataFrame | None:
    log_path = LOGS_DIR / "cartpole" / "training_log.npy"
    if not log_path.exists():
        return None
    data = np.load(log_path)
    df = pd.DataFrame(
        data,
        columns=["update", "mean_return", "policy_loss", "value_loss"],
    )
    return df


def evaluate_cartpole(model: ActorCritic, episodes: int, seed: int = 123) -> Tuple[float, float]:
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    returns: List[float] = []
    with torch.no_grad():
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            total = 0.0
            while not done:
                obs_tensor = torch.from_numpy(obs).float()
                logits, _ = model(obs_tensor)
                action = torch.argmax(logits).item()
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                done = terminated or truncated
            returns.append(total)
    env.close()
    returns_arr = np.array(returns, dtype=np.float32)
    return float(returns_arr.mean()), float(returns_arr.std())


@st.cache_data(show_spinner=False)
def load_frozenlake_log() -> pd.DataFrame | None:
    win_path = LOGS_DIR / "frozenlake" / "win_rate.npy"
    if not win_path.exists():
        return None
    data = np.load(win_path)
    df = pd.DataFrame(data, columns=["episode", "win_rate"])
    return df


@st.cache_data(show_spinner=False)
def load_frozenlake_q() -> np.ndarray | None:
    q_path = MODELS_DIR / "frozenlake_q.npy"
    if not q_path.exists():
        return None
    return np.load(q_path)


def evaluate_frozenlake(Q: np.ndarray, episodes: int, seed: int = 123) -> float:
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = int(np.argmax(Q[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if terminated and reward > 0:
                wins += 1
    env.close()
    return wins / max(episodes, 1)


@st.cache_data(show_spinner=False)
def load_accessions() -> List[str]:
    acc_path = DATA_DIR / "alphafold" / "ecoli_selected_accessions.txt"
    if not acc_path.exists():
        return []
    with open(acc_path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_pdb_ca(path: Path) -> np.ndarray:
    coords: List[Tuple[float, float, float]] = []
    if not path.exists():
        return np.empty((0, 3), dtype=np.float32)
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append((x, y, z))
    return np.array(coords, dtype=np.float32)


def load_toy_coords(acc: str) -> np.ndarray:
    toy_path = DATA_DIR / "mini_coords" / f"{acc}_coords.npy"
    if not toy_path.exists():
        return np.empty((0, 3, 3), dtype=np.float32)
    return np.load(toy_path)


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#181818")
    ax.spines["bottom"].set_color("#444")
    ax.spines["top"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["right"].set_color("#444")
    ax.tick_params(colors="#E5E5E5")
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontsize(10)


def _style_legend(leg: plt.Legend) -> None:
    leg.get_frame().set_facecolor("#1E1E1E")
    leg.get_frame().set_edgecolor("#333333")
    for text in leg.get_texts():
        text.set_color("#F5F5F5")


def plot_cartpole(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, facecolor="#101010")
    axes[0].plot(df["update"], df["mean_return"], label="Mean Return", color="#4CC9F0", linewidth=2)
    axes[0].axhline(475, color="#F72585", linestyle="--", linewidth=1.5, label="Target 475")
    axes[0].set_ylabel("Mean Return", color="#F5F5F5")
    axes[0].grid(alpha=0.15, color="#2A2A2A")
    _style_axis(axes[0])
    legend0 = axes[0].legend(loc="upper left")
    _style_legend(legend0)

    axes[1].plot(df["update"], df["policy_loss"], label="Policy Loss", color="#72EFDD", linewidth=2)
    axes[1].plot(df["update"], df["value_loss"], label="Value Loss", color="#FFA600", linewidth=1.8, alpha=0.85)
    axes[1].set_xlabel("Update", color="#F5F5F5")
    axes[1].set_ylabel("Loss", color="#F5F5F5")
    axes[1].grid(alpha=0.15, color="#2A2A2A")
    _style_axis(axes[1])
    legend1 = axes[1].legend(loc="upper left")
    _style_legend(legend1)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


def plot_frozenlake(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#101010")
    ax.plot(df["episode"], df["win_rate"], label="Win Rate", color="#4CC9F0", linewidth=2)
    ax.axhline(0.8, color="#F72585", linestyle="--", linewidth=1.5, label="Target 0.80")
    ax.set_xlabel("Episode", color="#F5F5F5")
    ax.set_ylabel("Win Rate", color="#F5F5F5")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.15, color="#2A2A2A")
    _style_axis(ax)
    legend = ax.legend(loc="lower right")
    _style_legend(legend)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


def plot_overlay(alpha_coords: np.ndarray, toy_coords: np.ndarray) -> Tuple[plt.Figure, float]:
    if alpha_coords.size == 0 or toy_coords.size == 0:
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="#101010")
        ax.set_facecolor("#181818")
        ax.text(0.5, 0.5, "Missing coordinates", ha="center", va="center")
        ax.axis("off")
        return fig, float("nan")
    toy_ca = toy_coords[:, 1, :]
    L = min(len(alpha_coords), len(toy_ca))
    if L == 0:
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="#101010")
        ax.set_facecolor("#181818")
        ax.text(0.5, 0.5, "Empty coordinates", ha="center", va="center")
        ax.axis("off")
        return fig, float("nan")
    alpha_xy = alpha_coords[:L, :2]
    toy_xy = toy_ca[:L, :2]
    deviations = np.linalg.norm(alpha_coords[:L] - toy_ca[:L], axis=1)
    mean_dev = float(deviations.mean())

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#101010")
    ax.set_facecolor("#181818")
    ax.plot(alpha_xy[:, 0], alpha_xy[:, 1], "-o", label="AlphaFold CA", color="#4CC9F0", markersize=4, linewidth=2)
    ax.plot(toy_xy[:, 0], toy_xy[:, 1], "-o", label="Toy CA", color="#FFA600", markersize=4, linewidth=2)
    ax.set_xlabel("X", color="#F5F5F5")
    ax.set_ylabel("Y", color="#F5F5F5")
    ax.set_title("Backbone XY Overlay", color="#F5F5F5")
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.15, color="#2A2A2A")
    _style_axis(ax)
    legend = ax.legend(loc="upper left")
    _style_legend(legend)
    fig.tight_layout()
    return fig, mean_dev


st.title("AlphaFold-nano")
st.markdown(
    """
    **Welcome!** This dashboard walks through a miniature end-to-end workflow:

    - **Reinforcement Learning.** We teach a tiny agent to balance the CartPole and solve the FrozenLake maze.
    - **Protein Toy Model.** We turn amino-acid sequences into a coarse backbone and compare it to AlphaFold predictions.

    Use the floating bar to explore the journey. Each view includes a quick primer so the visuals make sense even if you're new to the topic.
    """
)

with st.container():
    st.markdown('<div class="floating-nav">', unsafe_allow_html=True)
    section = st.session_state.get("section", "CartPole")
    dock_buttons = [("CartPole", ""), ("FrozenLake", ""), ("AlphaFold", "🧬")]
    cols = st.columns(len(dock_buttons))
    active_index = 1
    for idx, (label, emoji) in enumerate(dock_buttons, start=1):
        if section == label:
            active_index = idx
        with cols[idx - 1]:
            if st.button(f"{emoji}{label}", key=f"dock_{label}", use_container_width=False):
                st.session_state["section"] = label
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <style>
        .floating-nav div[data-testid="column"]:nth-of-type({active_index}) div[data-testid="stButton"] > button {{
            transform: translateY(-10px) scale(1.08);
            border-color: transparent;
            background: linear-gradient(135deg, #4CC9F0, #4361EE);
            box-shadow: 0px 28px 60px rgba(67, 97, 238, 0.45);
            color: #051937;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

section = st.session_state.get("section", "CartPole")

if section == "CartPole":
    st.header("CartPole: learning to balance")
    st.markdown(
        """
        **Goal:** Keep the pole upright for as long as possible. A perfect score is 500 steps.

        The agent watches the cart's position/velocity and the pole's angle, then decides to push left or right.
        """
    )

    log_df = load_cartpole_log()
    if log_df is None:
        st.warning("CartPole training log not found.")
    else:
        latest_return = float(log_df["mean_return"].iloc[-1])
        best_return = float(log_df["mean_return"].max())
        mean_policy_loss = float(log_df["policy_loss"].tail(50).mean())
        col1, col2, col3 = st.columns(3)
        col1.metric("Latest mean return", f"{latest_return:.1f}", delta=f"{latest_return - best_return:.1f}" if latest_return != best_return else None)
        col2.metric("Best mean return", f"{best_return:.1f}", help="Highest rolling score achieved during training (out of 500).")
        col3.metric("Recent policy loss", f"{mean_policy_loss:.3f}", help="Lower is better — indicates stable updates.")

        plot_cartpole(log_df)
        with st.expander("What am I looking at?", expanded=False):
            st.markdown(
                """
                - **Blue line:** Average score from the latest batch of training runs.
                - **Red dashed line:** The 475 target — above this, the agent is effectively perfect.
                - **Lower chart:** How hard the policy/value heads are working. Stable curves mean steady learning.
                """
            )

    model = load_cartpole_model()
    if model is None:
        st.warning("CartPole policy weights not found.")
    else:
        st.markdown(
            """
            ### Test the trained agent
            Pick how many trial runs to simulate. We replay the saved policy with greedy actions (no exploration).
            """
        )
        eval_eps = st.slider("Evaluation episodes", min_value=5, max_value=200, value=50, step=5)
        if st.button("Run CartPole evaluation", key="eval_cartpole"):
            mean_ret, std_ret = evaluate_cartpole(model, eval_eps)
            st.success(f"Average steps balanced: {mean_ret:.1f} ± {std_ret:.1f} (max 500)")
        with st.expander("How greedy evaluation works", expanded=False):
            st.markdown(
                """
                During training the agent explores randomly. Here we let it act deterministically
                — always choosing the highest-probability action — to see how well it performs when it
                simply trusts what it has learned.
                """
            )

elif section == "FrozenLake":
    st.header("FrozenLake: slippery maze solver")
    st.markdown(
        """
        **Goal:** Navigate an icy 8×8 grid to reach the goal `G` without falling into holes.

        It is a **slippery** environment, so even optimal choices can slide the agent in unwanted directions.
        """
    )
    frozen_df = load_frozenlake_log()
    if frozen_df is None:
        st.warning("FrozenLake win-rate log not found.")
    else:
        latest_win = float(frozen_df["win_rate"].iloc[-1])
        best_win = float(frozen_df["win_rate"].max())
        col1, col2 = st.columns(2)
        col1.metric("Latest win rate", f"{latest_win:.2%}", delta=f"{latest_win - best_win:.2%}" if latest_win != best_win else None)
        col2.metric("Best win rate", f"{best_win:.2%}", help="Peak sliding window win rate during training.")

        plot_frozenlake(frozen_df)
        with st.expander("Reading the curve", expanded=False):
            st.markdown(
                """
                - **Blue line:** Win rate averaged over the most recent batch of episodes.
                - **Red dashed line:** Project goal of 80% wins. Above it, the agent solves the level reliably.
                """
            )

    q_table = load_frozenlake_q()
    if q_table is None:
        st.warning("FrozenLake Q-table not found.")
    else:
        st.markdown(
            """
            ### Test the Q-table
            Choose how many evaluation games to run. We always pick the best action for the current state.
            """
        )
        eval_eps = st.slider("Evaluation episodes", min_value=100, max_value=2000, value=500, step=100, key="frozen_eval")
        if st.button("Run FrozenLake evaluation", key="eval_frozen"):
            win_rate = evaluate_frozenlake(q_table, eval_eps)
            st.success(f"Win rate over {eval_eps} episodes: {win_rate:.2%}")
        with st.expander("Why results vary", expanded=False):
            st.markdown(
                """
                FrozenLake remains stochastic even with a fixed policy. A single slip can send the agent into a hole,
                so we average over many games to get a reliable success rate.
                """
            )

else:
    st.header("AlphaFold-nano: toy backbone vs. AlphaFold")
    st.markdown(
        """
        **Goal:** Show how a toy sequence→structure model lines up with AlphaFold's predicted backbone.

        We focus on the **Cα trace** (one atom per residue) as a simple visual proxy for overall shape.
        """
    )
    accessions = load_accessions()
    if not accessions:
        st.warning("No accession list found. Run data prep scripts first.")
    else:
        acc = st.selectbox("Accession", accessions)
        if acc:
            pdb_path = DATA_DIR / "alphafold" / acc / f"{acc}.pdb"
            alpha_coords = load_pdb_ca(pdb_path)
            toy_coords = load_toy_coords(acc)
            fig, mean_dev = plot_overlay(alpha_coords, toy_coords)
            st.pyplot(fig)
            plt.close(fig)
            toy_len = toy_coords.shape[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("AlphaFold residues", len(alpha_coords))
            col2.metric("Toy residues", toy_len)
            col3.metric("Mean Cα deviation", f"{mean_dev:.2f} Å", help="Average distance between matching Cα atoms across the overlap region.")

            with st.expander("How to read the overlay", expanded=False):
                st.markdown(
                    """
                    - The **blue path** shows AlphaFold’s Cα coordinates projected onto the XY plane.
                    - The **orange path** is our toy model’s backbone. Because it is simplified, expect it to wobble more.
                    - The **mean deviation** summarizes how far apart the traces are on average (lower is better).
                    """
                )
            st.caption("Tip: Try switching accessions to see how the toy model handles different protein lengths.")
