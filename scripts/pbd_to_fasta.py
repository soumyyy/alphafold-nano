# filename: scripts/pdb_to_fasta.py

"""
Generate a minimal FASTA from AFDB PDB files when SEQRES is present.
Falls back to a dummy sequence ('A' repeated CA-count) if SEQRES is missing,
so you can still run the toy pipeline.

Usage:
    python scripts/pdb_to_fasta.py \
      --root /Users/soumya/Desktop/Projects/alphafold-nano/data/alphafold \
      --accessions-file /Users/soumya/Desktop/Projects/alphafold-nano/data/alphafold/ecoli_selected_accessions.txt \
      --out-fasta /Users/soumya/Desktop/Projects/alphafold-nano/data/sequences/ecoli_selected.fasta
"""

import argparse
import os

AA3_TO_1 = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I","LYS":"K","LEU":"L",
    "MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y"
}

def read_accessions(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def extract_seqres(pdb_path: str) -> str | None:
    residues = []
    with open(pdb_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("SEQRES"):
                tokens = line[19:].split()
                for tok in tokens:
                    if tok in AA3_TO_1:
                        residues.append(AA3_TO_1[tok])
    return "".join(residues) if residues else None

def count_ca(pdb_path: str) -> int:
    count = 0
    with open(pdb_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                count += 1
    return count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root folder containing per-accession PDBs.")
    parser.add_argument("--accessions-file", required=True, help="File with accessions to include.")
    parser.add_argument("--out-fasta", required=True, help="Output FASTA path.")
    args = parser.parse_args()

    accs = read_accessions(args.accessions_file)
    os.makedirs(os.path.dirname(args.out_fasta), exist_ok=True)

    written = 0
    with open(args.out_fasta, "w", encoding="utf-8") as out:
        for acc in accs:
            pdb_path = os.path.join(args.root, acc, f"{acc}.pdb")
            if not os.path.exists(pdb_path):
                continue
            seq = extract_seqres(pdb_path)
            if not seq:
                L = count_ca(pdb_path)
                if L <= 0:
                    continue
                seq = "A" * L  # dummy sequence for toy model input
            out.write(f">{acc}\n")
            for i in range(0, len(seq), 60):
                out.write(seq[i:i+60] + "\n")
            written += 1

    print(f"Wrote {written} FASTA entries to {args.out_fasta}")

if __name__ == "__main__":
    main()
