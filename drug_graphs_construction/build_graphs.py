import os
import argparse
import pandas as pd
import torch

from functions import smiles_to_graph


def sanitize_filename(s):
    s = str(s)
    keep = []
    for c in s:
        if c.isalnum() or c in ('-', '_'):
            keep.append(c)
    out = ''.join(keep)
    return out or s


def process_csv(input_csv, smiles_col='smiles', id_col='cid', out_dir='graphs', skip_invalid=True):
    df = pd.read_csv(input_csv)
    os.makedirs(out_dir, exist_ok=True)
    unique_df = df.drop_duplicates(subset=[id_col])

    for idx, row in df.iterrows():
        # Get ID (fall back to index if not present)
        if id_col in df.columns:
            id_val = row[id_col]
        else:
            id_val = idx

        # Get SMILES
        if smiles_col not in df.columns:
            raise ValueError(f"SMILES column '{smiles_col}' not found in CSV")
        smiles = row[smiles_col]
        if pd.isna(smiles) or str(smiles).strip() == "":
            if skip_invalid:
                print(f"Skipping ID {id_val}: empty SMILES")
                continue
            else:
                raise ValueError(f"Empty SMILES for ID {id_val}")

        data = smiles_to_graph(str(smiles))
        if data is None:
            if skip_invalid:
                print(f"Skipping ID {id_val}: invalid SMILES -> {smiles}")
                continue
            else:
                raise ValueError(f"Invalid SMILES for ID {id_val}: {smiles}")

        fname = sanitize_filename(id_val)
        out_path = os.path.join(out_dir, f"{fname}_drug_graph.pt")
        torch.save(data, out_path)

    print(f"Saved graphs to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Convert CSV SMILES to PyG .pt graphs')
    parser.add_argument('csv', help='Input CSV file')
    parser.add_argument('--smiles-col', default='smiles', help='Name of SMILES column (default: smiles)')
    parser.add_argument('--id-col', default='cid', help='Name of ID column (default: id). If missing, row index is used')
    parser.add_argument('--out-dir', default='drug_graphs_construction/graphs', help='Output directory for .pt files')
    parser.add_argument('--skip-invalid', action='store_true', help='Skip invalid or empty SMILES instead of failing')
    args = parser.parse_args()

    process_csv(args.csv, smiles_col=args.smiles_col, id_col=args.id_col, out_dir=args.out_dir, skip_invalid=args.skip_invalid)


if __name__ == '__main__':
    main()
