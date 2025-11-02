# alphafold-nano: Plan

## Objective
Build a minimal, CPU-only project that proves end-to-end reinforcement learning training on two small environments (CartPole-v1, FrozenLake-v1) and a stripped-down sequence→structure pipeline (“AlphaFold-nano”) that overlays toy backbone coordinates against AlphaFold PDB predictions from the E. coli bundle. Deliver reproducible scripts, model artifacts, and a Streamlit dashboard showing training curves, evaluations, and structure overlays.

## Scope
- RL: Train and evaluate CartPole-v1 (actor-critic) and FrozenLake-v1 (tabular Q-learning).
- Bio: Use E. coli AlphaFold bundle to obtain PDB predictions; generate minimal FASTA; run toy transformer to produce coarse backbone (N–CA–C) coordinates; visualize overlays and simple deviation metrics.
- Tooling: Python 3.10+, CPU-only PyTorch, Gymnasium, Streamlit, NumPy, Matplotlib. No GPUs. RAM budget < 1 GB.

## Data Sources and Acquisition
- AlphaFold DB E. coli bundle (compressed predictions, guaranteed coverage for small demo): [Escherichia coli — Download (456 MB)](https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000625_83333_ECOLI_v6.tar).
- UniProt human proteome “one protein per gene” FASTA (optional, already filtered earlier; not required for the E. coli track).
- Local path for raw E. coli bundle (example): `/Users/soumya/Desktop/Projects/alphafold-nano/data/UP000000625_83333_ECOLI_v6`.

## Data Preparation Workflow
1. Untar/unpack E. coli bundle into `data/alphafold/ecoli_raw`.
2. Select first N=10 PDB `.gz` files and decompress them to per-accession folders:
   - Script: `scripts/select_from_bundle.py`
   - Output:
     - `data/alphafold/<ACC>/<ACC>.pdb` for 10 accessions.
     - `data/alphafold/ecoli_selected_accessions.txt`.
3. Build a minimal FASTA from selected PDBs:
   - Script: `scripts/pdb_to_fasta.py` (reads SEQRES; falls back to dummy “A” repeated for CA-count).
   - Output: `data/sequences/ecoli_selected.fasta`.

## RL Components
### CartPole-v1 (Actor-Critic)
- Model: Shared MLP (64×64, ReLU) with policy logits (2 actions) and value head.
- Training: On-policy batches of 10 episodes/update; returns and advantages; entropy regularization 0.01; gradient clipping 0.5.
- Hyperparameters: lr=2e-3, γ=0.99, float32, CPU-only.
- Target: Mean return ≥ 475 over 100 eval episodes.
- Artifacts:
  - Weights: `models/cartpole_policy.pt`.
  - Logs: `logs/cartpole/training_log.npy` (updates, loss, mean return).

### FrozenLake-v1 (8×8, slippery=True) — Tabular Q-learning
- Table: `Q[s,a]` float32.
- Training: ε-greedy exploration with linear decay (1.0 → 0.05 over ~20k steps), α=0.1→0.01, γ=0.99.
- Target: Win rate ≥ 0.80 over 1000 eval episodes.
- Artifacts:
  - Q-table: `models/frozenlake_q.npy`.
  - Logs: `logs/frozenlake/win_rate.npy` (per 100 episodes).

## Bio Component: AlphaFold-nano Pipeline (Inference-only)
- Input: Short AA sequence from `ecoli_selected.fasta`.
- Features: AA embedding (20 letters), positional encoding inherent to transformer.
- Model: Tiny Transformer encoder (2 layers, d_model=128, nhead=4, feedforward=256).
- Head: Predict per-residue torsion angles (phi, psi).
- Projection: Fixed kinematics to N–CA–C backbone coordinates with approximate bond lengths; batch size 1; CPU-only; float32.
- Output: Per-accession `data/mini_coords/<ACC>_coords.npy` with shape `(L, 3, 3)`.

## Dashboard (Streamlit)
### Tabs
- RL — CartPole:
  - Plots: loss vs updates, mean return vs updates, target line at 475.
  - Interactive eval: run N episodes with greedy action, report mean ± std.
- RL — FrozenLake:
  - Plot: win rate vs episodes (per 100), target line at 0.80.
  - Interactive eval: run N episodes, report win rate.
- AlphaFold-nano:
  - Accession selector from `ecoli_selected_accessions.txt`.
  - Overlay plot: AlphaFold PDB Cα XY projection vs toy Cα XY projection.
  - Metric: Mean Cα deviation for first L residues (`L=min(len(toy), len(PDB_CA))`).
  - Optional: Show residue counts for both traces.

## Project Structure
```
alphafold-nano/
  data/
    UP000000625_83333_ECOLI_v6/                  # raw bundle (untar root)
    alphafold/
      ecoli_raw/                                 # unpacked files
      ecoli_selected_accessions.txt
      <ACC>/<ACC>.pdb                            # decompressed PDBs
    sequences/
      ecoli_selected.fasta
      human_subset.fasta                         # optional, from earlier step
  models/
    cartpole_policy.pt
    frozenlake_q.npy
  logs/
    cartpole/training_log.npy
    frozenlake/win_rate.npy
  scripts/
    select_from_bundle.py
    pdb_to_fasta.py
    train_cartpole.py
    train_frozenlake.py
    evaluate.py
  alphafold_mini/
    infer_backbone.py
  app/
    app.py                                       # unified Streamlit dashboard (3 tabs)
  requirements.txt
  PLAN.md                                        # this document
  README.md
```

## Requirements
- Python 3.10+
- `gymnasium`, `torch` (CPU), `numpy`, `pandas` (optional), `streamlit`, `matplotlib`
- Optional: `biopython` for FASTA/PDB convenience; project uses pure-Python fallbacks.

### requirements.txt
```
gymnasium
torch
numpy
pandas
streamlit
matplotlib
```

## Commands (Exact)
### Data
- Untar E. coli bundle:
```
mkdir -p data/alphafold/ecoli_raw
tar -xvf /Users/soumya/Desktop/Projects/alphafold-nano/data/UP000000625_83333_ECOLI_v6 \
  -C data/alphafold/ecoli_raw
```
- Select and decompress 10 PDBs:
```
python scripts/select_from_bundle.py \
  --bundle-dir data/alphafold/ecoli_raw \
  --outdir data/alphafold \
  --max 10
```
- Build minimal FASTA:
```
python scripts/pdb_to_fasta.py \
  --root data/alphafold \
  --accessions-file data/alphafold/ecoli_selected_accessions.txt \
  --out-fasta data/sequences/ecoli_selected.fasta
```

### RL Training
- CartPole:
```
python scripts/train_cartpole.py
```
- FrozenLake:
```
python scripts/train_frozenlake.py
```

### Toy Inference (Bio)
```
python alphafold_mini/infer_backbone.py \
  --fasta data/sequences/ecoli_selected.fasta \
  --out-dir data/mini_coords
```

### Dashboard
```
streamlit run app/app.py
```

## Success Criteria
- CartPole: mean return ≥ 475 over 100 eval episodes; stable curve and bounded gradients.
- FrozenLake: win rate ≥ 0.80 over 1000 eval episodes.
- Bio overlay: toy coordinates and AlphaFold PDB Cα traces render; deviation metric computed for first L residues; end-to-end pipeline runs on CPU within minutes per sequence.

## Resource Profile
- RAM peak: < 1 GB.
- CartPole training: 5–10 minutes CPU.
- FrozenLake training: 2–5 minutes CPU.
- Toy inference: seconds per short sequence.
- Streamlit: near-instant; interactions under a few seconds.

## Risks and Mitigations
- Missing SEQRES in AF PDB: fallback to dummy sequence length from CA count.
- Variable PDB residue counts vs sequence length: metrics restricted to first L residues; overlay remains informative.
- RL instability: use entropy coefficient 0.01, gradient clip 0.5, small learning rate; fixed seeds.
- Dataset volume: small bundle ensures quick extraction; avoid Swiss-Prot 27 GB bulk.

## Reproducibility
- Fixed seeds for NumPy/Torch/Gym.
- Float32 across pipelines.
- CPU-only runs; no nondeterministic GPU kernels.
- Logs and models saved consistently; evaluation scripts consume artifacts deterministically.

## App Integration Details (app.py)
- Tab “RL — CartPole”: load `logs/cartpole/training_log.npy`; plot loss and mean return; eval slider runs N episodes with greedy policy loaded from `models/cartpole_policy.pt`.
- Tab “RL — FrozenLake”: load `logs/frozenlake/win_rate.npy`; plot win rate; eval slider runs N episodes with greedy policy from `models/frozenlake_q.npy`.
- Tab “AlphaFold-nano”: read `data/alphafold/ecoli_selected_accessions.txt`; for selected accession:
  - PDB path: `data/alphafold/<ACC>/<ACC>.pdb`
  - Toy path: `data/mini_coords/<ACC>_coords.npy`
  - Parse PDB CA atoms; load toy coords; plot overlays; compute mean Cα deviation.

## Deliverables
- Scripts: data prep, RL training/eval, toy inference, Streamlit app.
- Artifacts: `cartpole_policy.pt`, `frozenlake_q.npy`, logs for both RL tasks, toy backbone `.npy` per selected accession.
- Dashboard: `app/app.py` rendering RL metrics and AlphaFold-nano overlays.
- Documentation: `PLAN.md`, `README.md` with commands and environment setup.

## Minimal Timeline
- Hour 1: Data unpack and selection; FASTA creation.
- Hour 2: CartPole training; checkpoint and logs.
- Hour 3: FrozenLake training; logs.
- Hour 4: Toy inference; coords generation.
- Hour 5: Streamlit integration; verification.

## Constraints and Boundaries
- Not reproducing AlphaFold training; inference-only toy for sequence→backbone with fixed kinematics.
- No external datasets beyond small E. coli bundle and minimal FASTA.
- CPU-only; keep models small; avoid heavy replay buffers or rendering during RL training.