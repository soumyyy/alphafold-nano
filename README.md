# alphafold-nano

CPU-only showcase that stitches together two reinforcement-learning benchmarks (CartPole, FrozenLake) with a toy sequence→structure pipeline (“AlphaFold-nano”). The Streamlit dashboard displays training progress, evaluation metrics, and an overlay between AlphaFold reference backbones and toy coordinates.

## Requirements
- Python 3.10+
- Dependencies listed in `requirements.txt`
- Local access to AlphaFold E. coli bundle (`data/UP000000625_83333_ECOLI_v6`)

Install packages into a virtual environment:
```bash
python3.11 -m venv .ven
source .ven/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Repo layout
- `scripts/select_from_bundle.py` – select N AlphaFold PDBs from the bundle
- `scripts/pdb_to_fasta.py` – build FASTA from PDB files
- `scripts/infer_backbone.py` – run toy transformer to produce backbone coordinates
- `rl_scripts/train_cartpole.py` – actor-critic trainer (saves model/logs/plot)
- `rl_scripts/train_frozenlake.py` – tabular Q-learning trainer
- `app/app.py` – Streamlit dashboard
- `data/` – inputs, selections, sequences, toy coords
- `logs/` – RL training logs and plots
- `models/` – trained policy/Q-table artifacts

## Data preparation
1. Extract the AlphaFold bundle: ~ will have to downlaod first
   ```bash
   mkdir -p data/alphafold/ecoli_raw
   tar -xvf data/UP000000625_83333_ECOLI_v6 -C data/alphafold/ecoli_raw
   ```
2. Select and decompress 10 accessions:
   ```bash
   python scripts/select_from_bundle.py \
     --bundle-dir data/alphafold/ecoli_raw \
     --outdir data/alphafold \
     --max 10
   ```
3. Build a FASTA for toy inference:
   ```bash
   python scripts/pdb_to_fasta.py \
     --root data/alphafold \
     --accessions-file data/alphafold/ecoli_selected_accessions.txt \
     --out-fasta data/sequences/ecoli_selected.fasta
   ```

## Reinforcement learning
CartPole and FrozenLake scripts are CPU-friendly; default seeds ensure deterministic artifacts.

Train CartPole (actor-critic):
```bash
python rl_scripts/train_cartpole.py
```
Defaults now run **1,000** update steps, collecting 15 episodes per update with a slightly smaller learning rate for a steadier critic.
Outputs:
- `models/cartpole_policy.pt`
- `logs/cartpole/training_log.npy`
- `logs/cartpole/training_plot.png`

Train FrozenLake (tabular Q-learning):
```bash
python rl_scripts/train_frozenlake.py
```
The configuration marches through **60,000** episodes with slower ε/α decay to explore longer before converging.
Outputs:
- `models/frozenlake_q.npy`
- `logs/frozenlake/win_rate.npy`
- `logs/frozenlake/training_plot.png`

## Toy backbone inference
Generate toy coordinates for selected accessions:
```bash
python scripts/infer_backbone.py \
  --fasta data/sequences/ecoli_selected.fasta \
  --out-dir data/mini_coords
```
Each accession produces `data/mini_coords/<ACC>_coords.npy` with shape `(L, 3, 3)` for atoms `[N, CA, C]`.

## Dashboard
Run the Streamlit application after artifacts exist:
```bash
streamlit run app/app.py
```
Tabs provide:
- **CartPole**: training curves, latest metrics, deterministic greedy evaluation
- **FrozenLake**: windowed win rates, current ε/α, greedy evaluation
- **AlphaFold-nano**: select accession, compare AlphaFold Cα trace with toy backbone, report mean Cα deviation

If necessary files are missing, the UI shows explicit error messages with the expected path.

## Troubleshooting
- Ensure all commands run inside the virtual environment so `gymnasium`, `torch`, and `streamlit` import correctly.
- Missing `data/alphafold/ecoli_selected_accessions.txt` or toy coordinate `.npy` files prevent the AlphaFold tab from rendering; rerun the data prep/inference scripts.
- Large FrozenLake value losses or low win rates usually indicate more training epochs or adjusted decay schedules are required.
