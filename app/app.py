#!/usr/bin/env python3
"""AlphaFold-nano dashboard (Streamlit).

Presents CartPole and FrozenLake RL training summaries plus a toy AlphaFold overlay.
All inputs are local artifacts; the app must never fail when files are missing.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn
from torch.distributions import Categorical

# ---------------------------------------------------------------------------
# Constants and styling
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

CARTPOLE_LOG_PATH = LOGS_DIR / "cartpole" / "training_log.npy"
CARTPOLE_MODEL_PATH = MODELS_DIR / "cartpole_policy.pt"
CARTPOLE_EVAL_SEED = 0

FROZENLAKE_LOG_PATH = LOGS_DIR / "frozenlake" / "win_rate.npy"
FROZENLAKE_Q_PATH = MODELS_DIR / "frozenlake_q.npy"
FROZEN_EVAL_SEED = 0

ACC_LIST_PATH = DATA_DIR / "alphafold" / "ecoli_selected_accessions.txt"
ALPHAFOLD_DIR = DATA_DIR / "alphafold"
TOY_COORDS_DIR = DATA_DIR / "mini_coords"

FROZEN_EPSILON_START = 1.0
FROZEN_EPSILON_END = 0.05
FROZEN_EPSILON_DECAY = 20_000
FROZEN_ALPHA_START = 0.1
FROZEN_ALPHA_END = 0.01
FROZEN_ALPHA_DECAY = 20_000

ACCENT_TOY = "tab:blue"
ACCENT_AF = "tab:orange"
TARGET_COLOR = "tab:red"

plt.style.use("default")
plt.rcParams.update(
    {
        "axes.facecolor": "#fff1dd",
        "figure.facecolor": "#fff1dd",
        "axes.edgecolor": "#443c32",
        "axes.labelcolor": "#2a261f",
        "xtick.color": "#2a261f",
        "ytick.color": "#2a261f",
        "legend.frameon": True,
        "legend.framealpha": 0.92,
    }
)

st.set_page_config(page_title="AlphaFold-nano Dashboard", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&display=swap');
    body, .stApp {
        background: radial-gradient(circle at top, #fff8ed 0%, #ffeeda 45%, #ffe5c7 100%);
        font-family: 'DM Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #2a261f;
    }
    .block-container {
        max-width: 900px;
        margin: auto;
        padding-top: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.55);
        border-radius: 999px;
        padding: 0.5rem 1.5rem;
        border: 1px solid rgba(78, 64, 46, 0.25);
        color: #5a5045;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f8b66f, #f58c4d);
        color: #2a261f !important;
        border-color: transparent;
        box-shadow: 0 10px 25px rgba(245, 140, 77, 0.25);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background: transparent !important;
    }
    .metric-label {font-size: 0.8rem; color: #6e6256; text-transform: uppercase; letter-spacing: 0.04em;}
    .metric-value {font-size: 1.6rem; font-weight: 700; color: #2a261f;}
    .metric-box {
        backdrop-filter: blur(14px);
        background: rgba(255, 255, 255, 0.55);
        border-radius: 24px;
        border: 1px solid rgba(90, 77, 63, 0.35);
        box-shadow: 0 18px 35px rgba(120, 94, 70, 0.18);
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.5rem;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1.1rem;
    }
    .metric-card {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.8), rgba(255, 244, 225, 0.5));
        border-radius: 18px;
        padding: 0.8rem 1rem;
        border: 1px solid rgba(244, 200, 150, 0.5);
    }
    h1, h2, h3, h4 {color: #2a261f; font-weight: 700;}
    .stSlider label {color: #4d4339;}
    .stSlider .css-1cpxqw2 {color: #4d4339;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """Matches the training architecture used for CartPole."""

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


# ---------------------------------------------------------------------------
# Helpers: IO with defensive error reporting
# ---------------------------------------------------------------------------


def _resolve(path: Path) -> str:
    return str(path.resolve())


def load_cartpole_log() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if not CARTPOLE_LOG_PATH.exists():
        return None, f"Missing CartPole log file: {_resolve(CARTPOLE_LOG_PATH)}"
    try:
        data = np.load(CARTPOLE_LOG_PATH)
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Failed to load { _resolve(CARTPOLE_LOG_PATH) }: {exc}"
    if data.ndim != 2 or data.shape[1] < 4:
        return None, f"Unexpected format in { _resolve(CARTPOLE_LOG_PATH) }"
    df = pd.DataFrame(
        data[:, :4],
        columns=["update", "mean_return", "policy_loss", "value_loss"],
    )
    return df, None


@st.cache_resource(show_spinner=False)
def load_cartpole_model() -> Tuple[Optional[ActorCritic], Optional[str]]:
    if not CARTPOLE_MODEL_PATH.exists():
        return None, f"Missing CartPole model file: {_resolve(CARTPOLE_MODEL_PATH)}"
    try:
        env = gym.make("CartPole-v1")
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        env.close()
    except Exception as exc:  # pragma: no cover - gym should be present
        return None, f"Gymnasium unavailable: {exc}"
    model = ActorCritic(obs_dim, n_actions=n_actions)
    try:
        state = torch.load(CARTPOLE_MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
    except Exception as exc:
        return None, f"Failed to load CartPole weights: {exc}"
    model.eval()
    return model, None


def load_frozenlake_log() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if not FROZENLAKE_LOG_PATH.exists():
        return None, f"Missing FrozenLake log file: {_resolve(FROZENLAKE_LOG_PATH)}"
    try:
        data = np.load(FROZENLAKE_LOG_PATH)
    except Exception as exc:
        return None, f"Failed to load { _resolve(FROZENLAKE_LOG_PATH) }: {exc}"
    if data.ndim != 2 or data.shape[1] < 2:
        return None, f"Unexpected format in { _resolve(FROZENLAKE_LOG_PATH) }"
    df = pd.DataFrame(data[:, :2], columns=["episode", "win_rate"])
    return df, None


@st.cache_resource(show_spinner=False)
def load_frozenlake_q() -> Tuple[Optional[np.ndarray], Optional[str]]:
    if not FROZENLAKE_Q_PATH.exists():
        return None, f"Missing FrozenLake Q-table: {_resolve(FROZENLAKE_Q_PATH)}"
    try:
        table = np.load(FROZENLAKE_Q_PATH)
    except Exception as exc:
        return None, f"Failed to load { _resolve(FROZENLAKE_Q_PATH) }: {exc}"
    return table, None


@st.cache_data(show_spinner=False)
def load_accessions() -> Tuple[Optional[list[str]], Optional[str]]:
    if not ACC_LIST_PATH.exists():
        return None, f"Missing accession list: {_resolve(ACC_LIST_PATH)}"
    try:
        with open(ACC_LIST_PATH, "r", encoding="utf-8") as handle:
            accs = [line.strip() for line in handle if line.strip()]
    except Exception as exc:
        return None, f"Failed to read { _resolve(ACC_LIST_PATH) }: {exc}"
    return accs, None


@st.cache_data(show_spinner=False)
def parse_pdb_ca(pdb_path: Path) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if not pdb_path.exists():
        return None, f"Missing AlphaFold PDB: {_resolve(pdb_path)}"
    coords: list[Tuple[float, float, float]] = []
    try:
        with open(pdb_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x, y, z))
    except Exception as exc:
        return None, f"Failed to parse { _resolve(pdb_path) }: {exc}"
    if not coords:
        return None, f"No Cα atoms found in { _resolve(pdb_path) }"
    return np.asarray(coords, dtype=np.float32), None


@st.cache_data(show_spinner=False)
def load_toy_coords(path: Path) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if not path.exists():
        return None, f"Missing toy coordinates: {_resolve(path)}"
    try:
        coords = np.load(path)
    except Exception as exc:
        return None, f"Failed to load { _resolve(path) }: {exc}"
    if coords.ndim != 3 or coords.shape[1:] != (3, 3):
        return None, f"Unexpected toy coordinate shape in { _resolve(path) }"
    return coords.astype(np.float32), None


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_cartpole(model: ActorCritic, episodes: int) -> Tuple[float, float, float]:
    start = time.perf_counter()
    env = gym.make("CartPole-v1")
    env.reset(seed=CARTPOLE_EVAL_SEED)
    env.action_space.seed(CARTPOLE_EVAL_SEED)
    returns: list[float] = []
    with torch.no_grad():
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            total = 0.0
            while not done:
                obs_tensor = torch.from_numpy(obs).float()
                logits, _ = model(obs_tensor)
                action = torch.argmax(logits).item()
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                done = terminated or truncated
            returns.append(total)
    env.close()
    arr = np.asarray(returns, dtype=np.float32)
    elapsed = time.perf_counter() - start
    return float(arr.mean()), float(arr.std()), elapsed


def evaluate_frozenlake(q_table: np.ndarray, episodes: int) -> Tuple[float, float]:
    start = time.perf_counter()
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    env.reset(seed=FROZEN_EVAL_SEED)
    wins = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if terminated and reward > 0:
                wins += 1
    env.close()
    elapsed = time.perf_counter() - start
    win_rate = wins / max(episodes, 1)
    return float(win_rate), elapsed


def linear_schedule(episode: float, start: float, end: float, decay_steps: float) -> float:
    fraction = min(max(episode, 0.0) / max(decay_steps, 1.0), 1.0)
    return start + fraction * (end - start)


def compute_ca_deviation(alpha_coords: np.ndarray, toy_coords: np.ndarray) -> float:
    toy_ca = toy_coords[:, 1, :]
    L = min(alpha_coords.shape[0], toy_ca.shape[0])
    if L == 0:
        return float("nan")
    diffs = toy_ca[:L] - alpha_coords[:L]
    dists = np.linalg.norm(diffs, axis=1)
    return float(dists.mean())


# ---------------------------------------------------------------------------
# Plot utilities
# ---------------------------------------------------------------------------


def plot_cartpole_training(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(df["update"], df["mean_return"], color="#333333", linewidth=2, label="Mean Return")
    axes[0].axhline(475, color=TARGET_COLOR, linestyle="--", linewidth=1.5, label="Target 475")
    axes[0].set_ylabel("Mean Return")
    axes[0].legend()

    axes[1].plot(df["update"], df["policy_loss"], color="#555555", linewidth=1.6, label="Policy Loss")
    axes[1].plot(df["update"], df["value_loss"], color="#888888", linewidth=1.6, label="Value Loss")
    axes[1].set_xlabel("Update")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_frozenlake_training(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df["episode"], df["win_rate"], color="#333333", linewidth=2, label="Win Rate")
    ax.axhline(0.80, color=TARGET_COLOR, linestyle="--", linewidth=1.5, label="Target 0.80")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_backbone_overlay(alpha_coords: np.ndarray, toy_coords: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    toy_ca = toy_coords[:, 1, :]
    L = min(alpha_coords.shape[0], toy_ca.shape[0])
    if L == 0:
        ax.text(0.5, 0.5, "No overlapping residues", ha="center", va="center")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
        return
    alpha_xy = alpha_coords[:L, :2]
    toy_xy = toy_ca[:L, :2]
    ax.plot(alpha_xy[:, 0], alpha_xy[:, 1], "-o", color=ACCENT_AF, linewidth=1.8, markersize=4, label="AlphaFold Cα")
    ax.plot(toy_xy[:, 0], toy_xy[:, 1], "-o", color=ACCENT_TOY, linewidth=1.8, markersize=4, label="Toy Cα")
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------------------


st.title("AlphaFold-nano")
st.subheader(
    "Reinforcement learning benchmarks and a sample sequence→structure overlay for *E. coli* accessions (with limited contraints)"
)

cartpole_tab, frozenlake_tab, alphafold_tab = st.tabs(["CartPole", "FrozenLake", "AlphaFold-nano"])

# CartPole tab ----------------------------------------------------------------
with cartpole_tab:
    log_df, log_err = load_cartpole_log()
    if log_err:
        st.error(log_err)
    else:
        latest_row = log_df.iloc[-1]
        best_mean = float(np.nanmax(log_df["mean_return"]))
        st.markdown(
            """
            <div class="metric-box">
              <div class="metric-grid">
                <div class="metric-card">
                  <div class="metric-label">Mean Return (latest)</div>
                  <div class="metric-value">{mean_latest:.1f}</div>
                </div>
                <div class="metric-card">
                  <div class="metric-label">Best Mean Return</div>
                  <div class="metric-value">{best_mean:.1f}</div>
                </div>
                <div class="metric-card">
                  <div class="metric-label">Updates Trained</div>
                  <div class="metric-value">{updates}</div>
                </div>
                <div class="metric-card">
                  <div class="metric-label">Policy Loss (latest)</div>
                  <div class="metric-value">{policy_loss:.4f}</div>
                </div>
                <div class="metric-card">
                  <div class="metric-label">Value Loss (latest)</div>
                  <div class="metric-value">{value_loss:.1f}</div>
                </div>
              </div>
            </div>
            """.format(
                mean_latest=latest_row["mean_return"],
                best_mean=best_mean,
                updates=int(latest_row["update"]),
                policy_loss=latest_row["policy_loss"],
                value_loss=latest_row["value_loss"],
            ),
            unsafe_allow_html=True,
        )

        st.markdown("#### Training history")
        plot_cartpole_training(log_df)

        model, model_err = load_cartpole_model()
        if model_err:
            st.error(model_err)
        else:
            st.markdown("#### Greedy evaluation")
            eval_eps = st.slider("Evaluation episodes", min_value=10, max_value=200, value=50, step=10)
            mean_ret, std_ret, elapsed = evaluate_cartpole(model, eval_eps)
            st.write(
                f"Mean return: **{mean_ret:.1f} ± {std_ret:.1f}** (episodes={eval_eps}, seed={CARTPOLE_EVAL_SEED}, "
                f"time={elapsed:.2f}s)"
            )

# FrozenLake tab --------------------------------------------------------------
with frozenlake_tab:
    frozen_df, frozen_err = load_frozenlake_log()
    if frozen_err:
        st.error(frozen_err)
    else:
        latest_row = frozen_df.iloc[-1]
        episode = float(latest_row["episode"])
        epsilon = linear_schedule(episode, FROZEN_EPSILON_START, FROZEN_EPSILON_END, FROZEN_EPSILON_DECAY)
        alpha = linear_schedule(episode, FROZEN_ALPHA_START, FROZEN_ALPHA_END, FROZEN_ALPHA_DECAY)

        col1, col2, col3 = st.columns(3)
        st.markdown(
            """
            <div class="metric-box">
              <div class="metric-grid">
                <div class="metric-card">
                  <div class="metric-label">Current Windowed Win Rate</div>
                  <div class="metric-value">{win_rate:.2f}</div>
                </div>
                <div class="metric-card">
                  <div class="metric-label">Episodes Trained</div>
                  <div class="metric-value">{episodes}</div>
                </div>
                <div class="metric-card">
                  <div class="metric-label">ε / α (current)</div>
                  <div class="metric-value">{eps:.2f} / {alpha_val:.2f}</div>
                </div>
              </div>
            </div>
            """.format(
                win_rate=latest_row["win_rate"],
                episodes=int(episode),
                eps=epsilon,
                alpha_val=alpha,
            ),
            unsafe_allow_html=True,
        )

        st.markdown("#### Training history")
        plot_frozenlake_training(frozen_df)

        q_table, q_err = load_frozenlake_q()
        if q_err:
            st.error(q_err)
        else:
            st.markdown("#### Greedy evaluation")
            eval_eps = st.slider("Evaluation episodes", min_value=100, max_value=5000, value=1000, step=100)
            win_rate, elapsed = evaluate_frozenlake(q_table, eval_eps)
            st.write(
                f"Win rate over {eval_eps} episodes: **{win_rate:.2%}** "
                f"(seed={FROZEN_EVAL_SEED}, time={elapsed:.2f}s). "
                "Slippery dynamics keep variance high even with a fixed table."
            )

# AlphaFold-nano tab ----------------------------------------------------------
with alphafold_tab:
    accessions, acc_err = load_accessions()
    if acc_err:
        st.error(acc_err)
    elif not accessions:
        st.warning("Accession list is empty.")
    else:
        st.markdown("Accessions derived from the AlphaFold *Escherichia coli* bundle.")
        accession = st.selectbox("Accession", options=accessions, index=0)
        pdb_path = ALPHAFOLD_DIR / accession / f"{accession}.pdb"
        toy_path = TOY_COORDS_DIR / f"{accession}_coords.npy"

        alpha_coords, alpha_err = parse_pdb_ca(pdb_path)
        toy_coords, toy_err = load_toy_coords(toy_path)

        if alpha_err:
            st.error(alpha_err)
        if toy_err:
            st.error(toy_err)

        if alpha_coords is not None and toy_coords is not None:
            mean_dev = compute_ca_deviation(alpha_coords, toy_coords)
            col1, col2, col3 = st.columns(3)
            st.markdown(
                """
                <div class="metric-box">
                  <div class="metric-grid">
                    <div class="metric-card">
                      <div class="metric-label">AlphaFold Cα residues</div>
                      <div class="metric-value">{af_residue}</div>
                    </div>
                    <div class="metric-card">
                      <div class="metric-label">Toy residues</div>
                      <div class="metric-value">{toy_residue}</div>
                    </div>
                    <div class="metric-card">
                      <div class="metric-label">Mean Cα deviation (Å)</div>
                      <div class="metric-value">{mean_dev:.2f}</div>
                    </div>
                  </div>
                </div>
                """.format(
                    af_residue=alpha_coords.shape[0],
                    toy_residue=toy_coords.shape[0],
                    mean_dev=mean_dev,
                ),
                unsafe_allow_html=True,
            )

            st.markdown("#### Backbone overlay")
            plot_backbone_overlay(alpha_coords, toy_coords)
