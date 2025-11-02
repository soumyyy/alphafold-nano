# filename: scripts/select_from_bundle.py

"""
Select N AlphaFold PDBs from an extracted bundle directory and decompress them
into per-accession folders for downstream use.

Usage:
    python scripts/select_from_bundle.py \
        --bundle-dir /Users/soumya/Desktop/Projects/alphafold-nano/data/UP000000625_83333_ECOLI_v6 \
        --outdir /Users/soumya/Desktop/Projects/alphafold-nano/data/alphafold \
        --max 10

Notes:
- Looks for files named like AF-<ACCESSION>-F1-model_v*.pdb.gz inside the bundle dir.
- Decompresses each into data/alphafold/<ACC>/<ACC>.pdb
- Writes data/alphafold/ecoli_selected_accessions.txt
"""

import argparse
import os
import gzip
import shutil
import re
from typing import List

def find_pdb_gz_files(root_dir: str) -> List[str]:
    files: List[str] = []
    for root, _, names in os.walk(root_dir):
        for n in names:
            if n.startswith("AF-") and n.endswith(".pdb.gz"):
                files.append(os.path.join(root, n))
    return sorted(files)

def parse_accession_from_filename(path: str) -> str | None:
    base = os.path.basename(path)
    m = re.match(r"AF-([A-Z0-9]+)-F\d+-model_v\d+\.pdb\.gz$", base)
    return m.group(1) if m else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True, help="Directory with AF-*-model_v*.pdb.gz files (untar output).")
    parser.add_argument("--outdir", required=True, help="Output root, e.g., data/alphafold.")
    parser.add_argument("--max", type=int, default=10, help="Number of PDBs to extract.")
    args = parser.parse_args()

    pdb_gz_files = find_pdb_gz_files(args.bundle_dir)
    if not pdb_gz_files:
        raise SystemExit(f"No PDB .gz files found in {args.bundle_dir}")

    os.makedirs(args.outdir, exist_ok=True)

    selected_accs: List[str] = []
    extracted = 0
    for gz_path in pdb_gz_files:
        acc = parse_accession_from_filename(gz_path)
        if not acc:
            continue

        acc_dir = os.path.join(args.outdir, acc)
        os.makedirs(acc_dir, exist_ok=True)
        dest_pdb = os.path.join(acc_dir, f"{acc}.pdb")

        if not os.path.exists(dest_pdb):
            # Decompress .gz -> .pdb
            with gzip.open(gz_path, "rb") as gz, open(dest_pdb, "wb") as out:
                shutil.copyfileobj(gz, out)

        selected_accs.append(acc)
        extracted += 1
        print(f"Extracted {acc} -> {dest_pdb}")

        if extracted >= args.max:
            break

    # Write accession list
    acc_list_path = os.path.join(args.outdir, "ecoli_selected_accessions.txt")
    with open(acc_list_path, "w", encoding="utf-8") as f:
        for acc in selected_accs:
            f.write(acc + "\n")

    print(f"Done: {extracted} PDBs extracted.")
    print(f"Accession list written to {acc_list_path}")

if __name__ == "__main__":
    main()
