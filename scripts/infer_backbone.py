#!/usr/bin/env python3
"""
Toy AlphaFold-nano backbone inference.

Reads sequences from a FASTA file, runs a tiny transformer encoder to predict
per-residue (phi, psi) torsion angles, and converts those angles into coarse
backbone coordinates (N, CA, C) that you can overlay against AlphaFold PDB
predictions. Designed for CPU-only inference.

Usage:
    python scripts/infer_backbone.py \
        --fasta data/sequences/ecoli_selected.fasta \
        --out-dir data/mini_coords
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX: Dict[str, int] = {aa: i for i, aa in enumerate(AA_ALPHABET)}
UNK_IDX = len(AA_ALPHABET)

@dataclass
class FastaEntry:
    header: str
    sequence: str

def parse_fasta(path: str) -> List[FastaEntry]:
    entries: List[FastaEntry] = []
    header: str | None = None
    seq_chunks: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    entries.append(FastaEntry(header, "".join(seq_chunks)))
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if header is not None:
        entries.append(FastaEntry(header, "".join(seq_chunks)))
    if not entries:
        raise SystemExit(f"No sequences read from FASTA: {path}")
    return entries

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)

class ToyBackboneModel(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 4, ff_dim: int = 256, num_layers: int = 2) -> None:
        super().__init__()
        vocab_size = len(AA_ALPHABET) + 1  # +1 for UNK
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=0.0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=False,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.torsion_head = nn.Linear(d_model, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (seq_len,) int64 tensor of residue indices.

        Returns:
            (seq_len, 2) tensor of torsion angles in radians (phi, psi).
        """
        x = self.embedding(tokens).unsqueeze(1)  # (seq, batch=1, d_model)
        x = self.pos_encoding(x)
        x = self.transformer(x).squeeze(1)  # (seq, d_model)
        torsions = torch.tanh(self.torsion_head(x)) * math.pi
        return torsions

def encode_sequence(seq: str) -> torch.Tensor:
    idxs = [
        AA_TO_IDX.get(aa, UNK_IDX)
        for aa in seq.upper()
        if aa.isalpha()
    ]
    if not idxs:
        raise ValueError("Sequence had no valid amino-acid characters.")
    return torch.tensor(idxs, dtype=torch.long)

def angles_to_direction(phi: float, psi: float) -> np.ndarray:
    x = math.cos(phi) * math.cos(psi)
    y = math.sin(phi) * math.cos(psi)
    z = math.sin(psi)
    vec = np.array([x, y, z], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return vec / norm

def build_backbone(phi: Sequence[float], psi: Sequence[float], step: float = 3.8) -> np.ndarray:
    """
    Convert torsion angles into crude backbone coordinates.

    Returns:
        coords: (L, 3, 3) array for atoms [N, CA, C] per residue.
    """
    length = len(phi)
    coords = np.zeros((length, 3, 3), dtype=np.float32)

    ca_position = np.zeros(3, dtype=np.float32)
    direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    for i in range(length):
        direction = angles_to_direction(phi[i], psi[i])
        if i > 0:
            ca_position = ca_position + step * direction

        perp = np.cross(direction, reference)
        if np.linalg.norm(perp) < 1e-3:
            perp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            perp = perp / (np.linalg.norm(perp) + 1e-8)
        binormal = np.cross(direction, perp)

        N = ca_position - 0.45 * step * direction + 0.25 * perp
        C = ca_position + 0.55 * step * direction - 0.25 * perp + 0.10 * binormal

        coords[i] = np.stack([N, ca_position.copy(), C], axis=0)

    return coords

def load_model(weights_path: str | None, device: torch.device) -> ToyBackboneModel:
    model = ToyBackboneModel()
    model.to(device)
    if weights_path:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
    model.eval()
    return model

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_coords(out_dir: str, accession: str, coords: np.ndarray) -> str:
    ensure_outdir(out_dir)
    out_path = os.path.join(out_dir, f"{accession}_coords.npy")
    np.save(out_path, coords)
    return out_path

def extract_accession(header: str) -> str:
    token = header.split()[0]
    if "|" in token:
        parts = token.split("|")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return token

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Infer toy backbone coordinates from FASTA sequences.")
    parser.add_argument("--fasta", required=True, help="Input FASTA with one or more sequences.")
    parser.add_argument("--out-dir", required=True, help="Directory to write <ACC>_coords.npy files.")
    parser.add_argument("--weights", help="Optional path to model weights (state_dict).")
    parser.add_argument("--device", default="cpu", help="Computation device, e.g. cpu or cuda.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for torch / numpy RNGs.")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    model = load_model(args.weights, device)

    entries = parse_fasta(args.fasta)
    print(f"Loaded {len(entries)} FASTA entries from {args.fasta}")

    for entry in entries:
        accession = extract_accession(entry.header)
        try:
            tokens = encode_sequence(entry.sequence)
        except ValueError as exc:
            print(f"[skip] {accession}: {exc}")
            continue

        with torch.no_grad():
            torsions = model(tokens.to(device))
        phi = torsions[:, 0].cpu().numpy()
        psi = torsions[:, 1].cpu().numpy()

        coords = build_backbone(phi, psi)
        dest = save_coords(args.out_dir, accession, coords)
        print(f"[ok] {accession}: length={coords.shape[0]} saved={dest}")

    print("Done.")

if __name__ == "__main__":
    main()
