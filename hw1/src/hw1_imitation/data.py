"""Dataset utilities for Push-T."""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

PUSHT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
ZARR_RELATIVE_PATH = Path("pusht") / "pusht_cchi_v7_replay.zarr"


@dataclass(frozen=True)
class Normalizer:
    """Feature-wise normalizer for states and actions."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @staticmethod
    # Prevent std = 0
    def _safe_std(std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return np.maximum(std, eps)

    @classmethod
    def from_data(cls, states: np.ndarray, actions: np.ndarray) -> "Normalizer":
        state_mean = states.mean(axis=0)
        state_std = cls._safe_std(states.std(axis=0))
        action_mean = actions.mean(axis=0)
        action_std = cls._safe_std(actions.std(axis=0))
        return cls(state_mean, state_std, action_mean, action_std)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return action * self.action_std + self.action_mean


def download_pusht(dataset_dir: Path) -> Path:
    """Download and extract the Push-T dataset if needed.

    Returns the path to the extracted Zarr dataset.
    """

    dataset_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = dataset_dir / ZARR_RELATIVE_PATH
    if zarr_path.exists():
        return zarr_path

    zip_path = dataset_dir / "pusht.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(PUSHT_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)

    return zarr_path


def load_pusht_zarr(zarr_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = zarr.open(zarr_path, mode="r")
    states = np.asarray(root["data"]["state"][:], dtype=np.float32)
    actions = np.asarray(root["data"]["action"][:], dtype=np.float32)
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    return states, actions, episode_ends


def build_valid_indices(episode_ends: np.ndarray, chunk_size: int) -> np.ndarray:
    starts = np.concatenate(([0], episode_ends[:-1]))
    indices: list[int] = []
    for start, end in zip(starts, episode_ends, strict=True):
        last_start = end - chunk_size
        if last_start < start:
            continue
        indices.extend(range(start, last_start + 1))
    return np.asarray(indices, dtype=np.int64)


class PushtChunkDataset(Dataset):
    """Dataset of (state, action_chunk) pairs using a sliding window."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        episode_ends: np.ndarray,
        chunk_size: int,
        normalizer: Normalizer | None = None,
    ) -> None:
        self.states = states
        self.actions = actions
        self.chunk_size = chunk_size
        self.normalizer = normalizer
        self.indices = build_valid_indices(episode_ends, chunk_size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = int(self.indices[idx])
        state = self.states[t]
        action_chunk = self.actions[t : t + self.chunk_size]

        if self.normalizer is not None:
            state = self.normalizer.normalize_state(state)
            action_chunk = self.normalizer.normalize_action(action_chunk)

        return (
            torch.from_numpy(state).float(),
            torch.from_numpy(action_chunk).float(),
        )


def visualize(
    states: np.ndarray,
    actions: np.ndarray,
    episode_ends: np.ndarray,
    out_dir: Path,
    chunk_size: int = 16,
) -> None:
    """Save overview plots of the Push-T data into out_dir."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    starts = np.concatenate(([0], episode_ends[:-1]))
    lengths = episode_ends - starts
    normalizer = Normalizer.from_data(states, actions)

    # 1. Agent path vs. action targets for the first 6 episodes.
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, ep in zip(axes.ravel(), range(6)):
        s, e = starts[ep], episode_ends[ep]
        ax.plot(states[s:e, 0], states[s:e, 1], c="tab:blue", label="agent pos")
        ax.plot(actions[s:e, 0], actions[s:e, 1], ".", c="tab:red", ms=3, label="action")
        ax.scatter(*states[s, :2], c="green", s=60, zorder=5, label="start")
        ax.scatter(*states[e - 1, :2], c="black", marker="x", s=60, zorder=5, label="end")
        ax.set_title(f"episode {ep} (len {e - s})")
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)
        ax.set_aspect("equal")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectories.png", dpi=110)

    # 2. State dimensions over time for episode 0.
    s, e = starts[0], episode_ends[0]
    names = ["agent x", "agent y", "block x", "block y", "block angle"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, name in enumerate(names):
        ax.plot(states[s:e, i], label=name)
    ax.set_xlabel("timestep")
    ax.set_title("episode 0: state over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "state_over_time.png", dpi=110)

    # 3. Raw vs. normalized distributions.
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0, 0].hist(states, bins=60, label=names)
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].set_title("raw states")
    axes[0, 1].hist(actions, bins=60, label=["a0", "a1"])
    axes[0, 1].legend()
    axes[0, 1].set_title("raw actions")
    axes[1, 0].hist(normalizer.normalize_state(states), bins=60)
    axes[1, 0].set_title("normalized states")
    axes[1, 1].hist(normalizer.normalize_action(actions), bins=60)
    axes[1, 1].set_title("normalized actions")
    fig.tight_layout()
    fig.savefig(out_dir / "distributions.png", dpi=110)

    # 4. Episode lengths and which start indices are valid for chunking.
    valid = np.zeros(len(states), dtype=bool)
    valid[build_valid_indices(episode_ends, chunk_size)] = True
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(lengths, bins=30)
    axes[0].set_title("episode lengths")
    axes[0].set_xlabel("steps")
    axes[1].plot(valid[: episode_ends[4]].astype(int), drawstyle="steps")
    for end in episode_ends[:5]:
        axes[1].axvline(end, c="r", ls="--")
    axes[1].set_title(f"valid start indices (chunk_size={chunk_size})")
    fig.tight_layout()
    fig.savefig(out_dir / "lengths_and_indices.png", dpi=110)

    # 5. One (state, action_chunk) sample as the model sees it.
    dataset = PushtChunkDataset(states, actions, episode_ends, chunk_size)
    t = int(dataset.indices[50])
    chunk = actions[t : t + chunk_size]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(states[t : t + chunk_size, 0], states[t : t + chunk_size, 1], "-o", label="agent")
    ax.plot(chunk[:, 0], chunk[:, 1], "-x", c="tab:red", label="action chunk")
    ax.scatter(*states[t, :2], c="green", s=80, zorder=5, label="state at t")
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    ax.set_aspect("equal")
    ax.legend(fontsize=7)
    ax.set_title(f"sample t={t}, chunk_size={chunk_size}")
    fig.tight_layout()
    fig.savefig(out_dir / "one_sample.png", dpi=110)


if __name__ == "__main__":
    data_path = download_pusht(Path("data"))
    states, actions, episode_ends = load_pusht_zarr(data_path)
    lengths = np.diff(episode_ends, prepend=0)

    np.set_printoptions(precision=2, suppress=True, linewidth=120)
    print(f"states:   {states.shape} {states.dtype}")
    print(f"actions:  {actions.shape} {actions.dtype}")
    print(
        f"episodes: {len(episode_ends)} "
        f"(len min/mean/max = {lengths.min()}/{lengths.mean():.1f}/{lengths.max()})"
    )
    print("state  mean:", states.mean(axis=0))
    print("state  std: ", states.std(axis=0))
    print("action mean:", actions.mean(axis=0))
    print("action std: ", actions.std(axis=0))
    for k in (1, 8, 16):
        print(f"chunk_size={k}: {len(build_valid_indices(episode_ends, k))} samples")

    out_dir = Path("figures")
    visualize(states, actions, episode_ends, out_dir)
    print(f"Saved plots to {out_dir.resolve()}")

