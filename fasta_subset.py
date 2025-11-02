# filename: fasta_subset.py
"""
Read a FASTA file, filter sequences by length and simple header heuristics,
and write a small subset FASTA plus a UniProt accession list.

Usage (Biopython mode):
    python fasta_subset.py --input data/UP000005640_9606.fasta \
                           --out-fasta data/sequences/human_subset.fasta \
                           --out-acc data/sequences/human_subset_accessions.txt \
                           --min-len 80 --max-len 200 --max 10 --mode biopython

Usage (pure-Python mode):
    python fasta_subset.py --input data/UP000005640_9606.fasta \
                           --out-fasta data/sequences/human_subset.fasta \
                           --out-acc data/sequences/human_subset_accessions.txt \
                           --min-len 80 --max-len 200 --max 10 --mode pure
"""

import argparse
import os
import sys
from typing import List, Tuple

EXCLUDED_TERMS = [
    "transmembrane", "signal peptide", "secreted", "gpi-anchored",
    "glycosylphosphatidylinositol", "lipoprotein"
]

def header_has_excluded_terms(header: str) -> bool:
    """Heuristic exclusion based on header phrases to avoid membrane/signal peptides."""
    h = header.lower()
    return any(term in h for term in EXCLUDED_TERMS)

def extract_uniprot_accession_from_header(header: str) -> str:
    """
    Extract a UniProt accession from common FASTA header formats.
    Examples:
      >sp|P12345|PROT_HUMAN ...
      >tr|Q9ABC1|...
      >A0A123ABC ...
    Fallback: first whitespace-delimited token.
    """
    first_token = header.split()[0]
    if "|" in first_token:
        parts = first_token.split("|")
        if len(parts) >= 2:
            return parts[1]
    return first_token

def write_fasta(entries: List[Tuple[str, str, str]], out_path: str) -> None:
    """Write selected entries to FASTA."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for acc, header, seq in entries:
            f.write(f">{header}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")

def write_accessions(entries: List[Tuple[str, str, str]], out_path: str) -> None:
    """Write UniProt accessions to a text file, one per line."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for acc, _, _ in entries:
            f.write(acc + "\n")

def parse_fasta_biopython(path: str) -> List[Tuple[str, str]]:
    """Parse FASTA using Biopython and return list of (header, sequence)."""
    try:
        from Bio import SeqIO  # pip install biopython
    except ImportError:
        print("Biopython not installed. Install with: pip install biopython", file=sys.stderr)
        raise
    records = list(SeqIO.parse(path, "fasta"))
    if not records:
        raise ValueError(f"No sequences found in FASTA: {path}")
    return [(rec.description, str(rec.seq)) for rec in records]

def parse_fasta_pure(path: str) -> List[Tuple[str, str]]:
    """Minimal FASTA reader without external libraries. Returns list of (header, sequence)."""
    entries: List[Tuple[str, str]] = []
    header = None
    seq_chunks: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    entries.append((header, "".join(seq_chunks)))
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if header is not None:
        entries.append((header, "".join(seq_chunks)))
    if not entries:
        raise ValueError(f"No sequences found in FASTA: {path}")
    return entries

def filter_sequences(
    entries: List[Tuple[str, str]],
    min_len: int,
    max_len: int,
    max_keep: int,
    exclude_membrane_like: bool = True
) -> List[Tuple[str, str, str]]:
    """Filter by length and optional header exclusion. Returns (accession, header, sequence)."""
    kept: List[Tuple[str, str, str]] = []
    for header, seq in entries:
        L = len(seq)
        if L < min_len or L > max_len:
            continue
        if exclude_membrane_like and header_has_excluded_terms(header):
            continue
        acc = extract_uniprot_accession_from_header(header)
        kept.append((acc, header, seq))
        if len(kept) >= max_keep:
            break
    return kept

def main():
    parser = argparse.ArgumentParser(description="Subset a FASTA to short, non-membrane proteins and export accessions.")
    parser.add_argument("--input", required=True, help="Path to input FASTA (e.g., data/UP000005640_9606.fasta).")
    parser.add_argument("--out-fasta", required=True, help="Path to write filtered FASTA (e.g., data/sequences/human_subset.fasta).")
    parser.add_argument("--out-acc", required=True, help="Path to write accession list (e.g., data/sequences/human_subset_accessions.txt).")
    parser.add_argument("--min-len", type=int, default=80, help="Minimum sequence length to keep.")
    parser.add_argument("--max-len", type=int, default=200, help="Maximum sequence length to keep.")
    parser.add_argument("--max", type=int, default=10, help="Maximum number of sequences to keep.")
    parser.add_argument("--mode", choices=["biopython", "pure"], default="biopython", help="Parser mode.")
    parser.add_argument("--include-membrane", action="store_true", help="Include sequences even if headers mention transmembrane/signal.")
    args = parser.parse_args()

    # Parse
    if args.mode == "biopython":
        try:
            entries = parse_fasta_biopython(args.input)
        except ImportError:
            print("Biopython not installed; falling back to pure-Python parser.", file=sys.stderr)
            entries = parse_fasta_pure(args.input)
    else:
        entries = parse_fasta_pure(args.input)

    # Filter
    selected = filter_sequences(
        entries,
        min_len=args.min_len,
        max_len=args.max_len,
        max_keep=args.max,
        exclude_membrane_like=not args.include_membrane
    )
    if not selected:
        raise SystemExit("No sequences matched filters. Try relaxing length bounds or allowing membrane entries.")

    # Write outputs
    write_fasta(selected, args.out_fasta)
    write_accessions(selected, args.out_acc)

    # Summary
    print(f"Wrote {len(selected)} sequences to {args.out_fasta}")
    print(f"Wrote UniProt accessions to {args.out_acc}")
    for acc, header, seq in selected[:5]:
        print(f"- {acc}\tlen={len(seq)}\theader={header[:60]}...")

if __name__ == "__main__":
    main()
