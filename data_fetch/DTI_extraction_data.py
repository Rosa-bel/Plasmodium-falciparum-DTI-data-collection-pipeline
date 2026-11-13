"""
Expand P. falciparum DTI dataset beyond ChEMBL (~600 rows) toward 1500+ rows.

Sources added:
  1. PubChem BioAssay  (via PUG REST, target = taxonomy ID 5833 or specific UniProt/gene)
  2. BindingDB         (via their bulk TSV-by-target download endpoint)

Output: a single harmonized CSV compatible with your existing ChEMBL schema:
    drug_inchikey, drug_smiles, target_uniprot, target_gene_id,
    activity_type, activity_value_nM, pActivity,
    activity_label (active / inactive / inconclusive),
    source, confidence_flag

Every row is guaranteed to carry drug_smiles plus an interaction indicator:
  - activity_label: categorical active/inactive/inconclusive (derived from
    PubChem's PUBCHEM_ACTIVITY_OUTCOME, or a cutoff-based rule for BindingDB)
  - pActivity: quantitative -log10(M) value when a numeric IC50/Ki/Kd/EC50
    is available (NaN if the source only reported a qualitative call)

Run this LOCALLY (not in a sandboxed/restricted-network env) since it hits
pubchem.ncbi.nlm.nih.gov and bindingdb.org directly.

Usage:
    python expand_dti_dataset.py --targets targets.csv --chembl chembl_extract.csv --out merged_dti.csv

Where targets.csv has at least columns: Gene_ID, UniProt_ID, Product
(this should just be your existing 37-row target list, with a UniProt_ID column
added if it isn't already there).
"""

import argparse
import os
import re
import time
import requests
import pandas as pd
from rdkit import Chem
import pubchempy as pcp
from pubchempy import Compound
from rdkit.Chem import inchi

PFALCIPARUM_TAXID = 5833


def normalize_text(value):
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def infer_target_from_description(description, protein_df):
    desc = normalize_text(description)
    if not desc:
        return None, None

    best_name, best_entry, best_score = None, None, -1
    for _, row in protein_df.iterrows():
        protein_name = None
        for col in ["Protein names", "Protein_Name", "Protein name", "Protein Names", "protein_name"]:
            if col in row.index:
                protein_name = row[col]
                break
        if pd.isna(protein_name):
            continue

        norm_name = normalize_text(protein_name)
        if not norm_name:
            continue

        score = 0
        if norm_name in desc:
            score += 20

        tokens = set(norm_name.split())
        desc_tokens = set(desc.split())
        overlap = len(tokens & desc_tokens)
        score += overlap * 3
        if overlap >= 2:
            score += 5

        if score > best_score:
            best_name = protein_name
            best_entry = row.get("Entry") or row.get("UniProt_ID")
            best_score = score

    if best_score >= 6:
        return best_name, best_entry
    return None, None


def extract_interaction_tests(target_path, chembl_path, out_path):
    protein_df = pd.read_csv(target_path)
    chembl_df = pd.read_csv(chembl_path, sep=";")

    if "Description" not in chembl_df.columns:
        raise ValueError("The ChEMBL input must contain a Description column")

    relevant = chembl_df.copy()
    relevant["Description"] = relevant["Description"].fillna("")

    if "Confidence Score" in relevant.columns:
        relevant["Confidence Score"] = pd.to_numeric(relevant["Confidence Score"], errors="coerce")

    filter_mask = relevant["Description"].astype(str).str.len() > 0
    if "BAO Format" in relevant.columns:
        filter_mask &= relevant["BAO Format"].astype(str).str.lower().str.contains("single protein", na=False)
    if "Confidence Score" in relevant.columns:
        filter_mask &= relevant["Confidence Score"].ge(8)

    relevant = relevant.loc[filter_mask].copy()

    matched_names = []
    matched_entries = []
    for description in relevant["Description"]:
        name, entry = infer_target_from_description(description, protein_df)
        matched_names.append(name)
        matched_entries.append(entry)

    relevant["target_name"] = matched_names
    relevant["target_uniprot"] = matched_entries
    relevant["interaction_test"] = True

    out_df = pd.DataFrame({
        "assay_chembl_id": relevant.get("Assay ChEMBL ID"),
        "assay_type": relevant.get("Assay Type"),
        "description": relevant.get("Description"),
        "organism": relevant.get("Organism"),
        "target_chembl_id": relevant.get("Target ChEMBL ID"),
        "target_name": relevant.get("target_name"),
        "target_uniprot": relevant.get("target_uniprot"),
        "confidence_score": relevant.get("Confidence Score"),
        "confidence_label": relevant.get("Confidence Label"),
        "source": relevant.get("Source"),
        "strain": relevant.get("Strain"),
        "variant_sequence_accession": relevant.get("Variant Sequence Accession"),
        "variant_sequence_mutation": relevant.get("Variant Sequence Mutation"),
        "interaction_test": relevant.get("interaction_test"),
    })

    out_df = out_df.drop_duplicates(
        subset=["assay_chembl_id", "target_chembl_id", "description"], keep="first"
    ).reset_index(drop=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} interaction tests to {out_path}")
    return out_df

# ---------------------------------------------------------------------------
# 1. PubChem BioAssay
# ---------------------------------------------------------------------------

def pubchem_assays_for_taxid(taxid=PFALCIPARUM_TAXID, max_aids=100):
    """Get assay IDs (AIDs) associated with a taxonomy ID (e.g., P. falciparum)."""
    
    # Correct domain is 'taxonomy', not 'assay/target'
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/taxonomy/taxid/{taxid}/aids/JSON"
    
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except requests.RequestException as exc:
        print(f"[PubChem] could not fetch assay IDs for taxid {taxid}: {exc}")
        return []

    try:
        data = r.json()
    except ValueError as exc:
        print(f"[PubChem] could not parse assay response for taxid {taxid}: {exc}")
        return []

    # PubChem returns ID lists inside 'IdentifierList'
    aids = data["InformationList"]["Information"][0]["AID"]
    
    print(f"[PubChem] Found {len(aids)} assays for taxid {taxid}.")
    return aids[:max_aids]


def pubchem_id_to_smiles(identifier, id_type="cid", timeout=30):
    """Resolve a PubChem compound or substance identifier to a canonical SMILES."""
    if pd.isna(identifier):
        return None

    try:
        identifier = str(int(float(identifier)))
    except Exception:
        identifier = str(identifier).strip()

    if not identifier:
        return None

    endpoint = "sid" if str(id_type).lower() == "sid" else "cid"
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{endpoint}/{identifier}"
        "/property/CanonicalSMILES/JSON"
    )
    try:
        r = requests.get(url, timeout=timeout)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        props = r.json().get("PropertyTable", {}).get("Properties", [])
    except Exception:
        return None
    if not props:
        return None
    return props[0].get("CanonicalSMILES")


def pubchem_ids_to_smiles(ids, id_type="cid", sleep=0.3):
    """Resolve PubChem identifiers (CID or SID) to canonical SMILES."""
    id_to_smiles = {}
    ids = [c for c in pd.unique(pd.Series(ids).dropna())]
    for identifier in ids:
        # pubchempy's request() does `",".join(str(x) for x in identifier)`
        # when given something it doesn't recognize as a plain scalar --
        # numpy.float64 / numpy.int64 break that join, so force a plain
        # Python int (or str fallback) before calling out.
        try:
            clean_id = int(identifier)
        except (TypeError, ValueError):
            clean_id = str(identifier).strip()
            if not clean_id:
                continue

        print("identifier:", clean_id)
        smiles = None
        try:
            if str(id_type).lower() == "cid":
                compound = Compound.from_cid(clean_id)
                smiles = compound.canonical_smiles
            else:
                substances = pcp.Substance.from_sid(clean_id)
                if not isinstance(substances, list):
                    substances = [substances]
                for s in substances:
                    cids = getattr(s, "cids", None)
                    if cids:
                        compound = Compound.from_cid(cids[0])
                        smiles = compound.canonical_smiles
                        break
        except Exception as exc:
            print(f"[PubChem] lookup failed for {id_type.upper()} {clean_id}: {exc}")

        if smiles:
            id_to_smiles[identifier] = smiles
        time.sleep(sleep)
    return id_to_smiles

def pubchem_assays_for_accession(uniprot_accession, max_aids=200, sleep=0.2):
    """Get AIDs where this specific UniProt accession is the registered target."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/target/accession/{uniprot_accession}/aids/JSON"
    try:
        r = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        print(f"[PubChem] accession lookup failed for {uniprot_accession}: {exc}")
        return []
    time.sleep(sleep)
    if r.status_code != 200:
        return []
    try:
        data = r.json()
        aids = data["InformationList"]["Information"][0]["AID"]
    except (KeyError, IndexError, ValueError):
        return []
    return aids[:max_aids]


def collect_pubchem_per_target(target_df, max_aids_per_target=200, sleep=0.3, stats_rows=None):
    """Loop over your target list, query PubChem per-accession, tag every
    resulting row with target_uniprot/target_gene_id directly -- no
    description-parsing or fuzzy matching needed."""
    all_rows = []
    for _, row in target_df.dropna(subset=["Entry"]).iterrows():
        uid = row["Entry"]
        aids = pubchem_assays_for_accession(uid, max_aids=max_aids_per_target, sleep=sleep)
        print(f"[PubChem] {uid} ({row.get('Gene_ID')}): {len(aids)} assays")

        target_rows = []
        for aid in aids:
            df = pubchem_assay_results(aid)  # your existing function, unchanged
            time.sleep(sleep)
            if df is None or df.empty:
                continue

            sub = pd.DataFrame()
            smiles_col = next(
                (c for c in df.columns
                 if c.lower() in {"pubchem_ext_datasource_smiles", "smiles", "canonical_smiles"}),
                None,
            )
            sub["drug_smiles"] = df[smiles_col] if smiles_col is not None else pd.NA
            sub["pubchem_cid"] = pd.to_numeric(df.get("PUBCHEM_CID"), errors="coerce")
            sub["activity_outcome"] = df.get("PUBCHEM_ACTIVITY_OUTCOME")
            sub["activity_type"] = df.get("Standard Type")
            sub["activity_value_nM"] = pd.to_numeric(df.get("Standard Value"), errors="coerce")
            sub["aid"] = aid

            # Tagged immediately -- this is the whole point
            sub["target_uniprot"] = uid
            sub["target_gene_id"] = row.get("Gene_ID")

            target_rows.append(sub)

        if not target_rows:
            if stats_rows is not None:
                stats_rows.append({
                    "source": "PubChem", "target_uniprot": uid,
                    "target_gene_id": row.get("Gene_ID"), "target_name": row.get("Common_Name"),
                    "assays_requested": len(aids), "assays_returned": 0,
                    "ligands_found": 0, "ligands_with_smiles": 0,
                    "ligands_with_activity_value": 0, "active_count": 0, "status": "empty",
                })
            continue

        target_df_out = pd.concat(target_rows, ignore_index=True)
        all_rows.append(target_df_out)

        if stats_rows is not None:
            stats_rows.append({
                "source": "PubChem", "target_uniprot": uid,
                "target_gene_id": row.get("Gene_ID"), "target_name": row.get("Common_Name"),
                "assays_requested": len(aids), "assays_returned": target_df_out["aid"].nunique(),
                "ligands_found": len(target_df_out),
                "ligands_with_smiles": int(target_df_out["drug_smiles"].notna().sum()),
                "ligands_with_activity_value": int(target_df_out["activity_value_nM"].notna().sum()),
                "active_count": 0, "status": "ok",
            })

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)

# ---------------------------------------------------------------------------
# 2. BindingDB
# ---------------------------------------------------------------------------

def bindingdb_by_uniprot(uniprot_id, cutoff_nm=1000000):
    """
    BindingDB exposes a REST endpoint for ligands by UniProt target.
    Returns a dataframe of ligand SMILES + affinity values.
    """
    print(f"[BindingDB] Fetching ligands for UniProt ID: {uniprot_id}")
    url = (
        "https://bindingdb.org/rest/getLigandsByUniprot/"
        f"?uniprot={uniprot_id}&cutoff={cutoff_nm}&response=application/json"
    )
    try:
        r = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        print(f"[BindingDB] request failed for {uniprot_id}: {exc}")
        return pd.DataFrame()
    if r.status_code != 200:
        return pd.DataFrame()
    try:
        data = r.json()
    except Exception:
        return pd.DataFrame()

    hits = data.get("getLigandsByUniprotResponse", {}).get("affinities", [])
    if not hits:
        return pd.DataFrame()
    df = pd.json_normalize(hits)
    print("Number of hits:", len(df))
    df["target_uniprot"] = uniprot_id
    return df


def collect_bindingdb(target_df, sleep=0.5, active_cutoff_nm=1000000, stats_rows=None):
    """target_df must have a UniProt_ID column.

    active_cutoff_nm: affinity threshold (nM) below which a row is labeled
    'active'. BindingDB gives no categorical call by default, so we derive
    one -- 10 microM is a common, conservative default for phenotypic-style
    cutoffs; tighten to 1000 nM if you want a stricter definition matching
    your ChEMBL IC50/Ki/Kd selection.
    """
    all_rows = []
    for _, row in target_df.dropna(subset=["Entry"]).iterrows():
        uid = row["Entry"]
        df = bindingdb_by_uniprot(uid)
        time.sleep(sleep)
        if df.empty:
            if stats_rows is not None:
                stats_rows.append({
                    "source": "BindingDB",
                    "target_uniprot": uid,
                    "target_gene_id": row.get("Gene_ID"),
                    "target_name": row.get("Protein names") or row.get("Protein_Name") or row.get("Protein name"),
                    "assays_requested": 1,
                    "assays_returned": 0,
                    "ligands_found": 0,
                    "ligands_with_smiles": 0,
                    "ligands_with_activity_value": 0,
                    "active_count": 0,
                    "status": "empty",
                })
            continue

        df["target_gene_id"] = row.get("Gene_ID")
        all_rows.append(df)
        print(f"[BindingDB] {uid}: {len(df)} ligand affinities")

        smiles_col = next(
            (c for c in df.columns if c.lower() in ("smiles", "ligand_smiles", "monomerid.smiles")),
            None,
        )
        affinity_col = next(
            (c for c in df.columns if c.lower() in ("affinity", "value", "ki", "ic50", "kd", "ec50")),
            None,
        )
        df_local = df.copy()
        df_local["drug_smiles"] = df_local[smiles_col] if smiles_col else None
        df_local["activity_value_nM"] = pd.to_numeric(df_local[affinity_col], errors="coerce") if affinity_col else pd.NA
        active_count = int((df_local["activity_value_nM"].notna() & (df_local["activity_value_nM"] <= active_cutoff_nm)).sum()) if "activity_value_nM" in df_local.columns else 0

        if stats_rows is not None:
            stats_rows.append({
                "source": "BindingDB",
                "target_uniprot": uid,
                "target_gene_id": row.get("Gene_ID"),
                "target_name": row.get("Protein names") or row.get("Protein_Name") or row.get("Protein name"),
                "assays_requested": 1,
                "assays_returned": 1,
                "ligands_found": int(len(df_local)),
                "ligands_with_smiles": int(df_local["drug_smiles"].notna().sum()),
                "ligands_with_activity_value": int(df_local["activity_value_nM"].notna().sum()),
                "active_count": active_count,
                "status": "ok",
            })

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)

    # BindingDB JSON typically includes a SMILES field per ligand; the exact
    # key name varies by endpoint version, so check a few common candidates.
    smiles_col = next(
        (c for c in out.columns if c.lower() in ("smiles", "ligand_smiles", "monomerid.smiles")),
        None,
    )
    out["drug_smiles"] = out[smiles_col] if smiles_col else None

    # Affinity value column also varies (affinity, ki, ic50, kd, ec50...) --
    # take whichever numeric one is present per row.
    affinity_col = next(
        (c for c in out.columns if c.lower() in ("affinity", "value", "ki", "ic50", "kd", "ec50")),
        None,
    )
    out["activity_value_nM"] = pd.to_numeric(out[affinity_col], errors="coerce") if affinity_col else pd.NA

    out["activity_label"] = out["activity_value_nM"].apply(
        lambda v: "active" if pd.notna(v) and v <= active_cutoff_nm
        else ("inactive" if pd.notna(v) else "inconclusive")
    )

    return out


def save_fetch_stats(stats_rows, out_path):
    stats_columns = [
        "source",
        "target_uniprot",
        "target_gene_id",
        "target_name",
        "assays_requested",
        "assays_returned",
        "ligands_found",
        "ligands_with_smiles",
        "ligands_with_activity_value",
        "active_count",
        "status",
    ]
    if not stats_rows:
        pd.DataFrame(columns=stats_columns).to_csv(out_path, index=False)
        print(f"Wrote empty fetch statistics file to {out_path}")
        return

    stats_df = pd.DataFrame(stats_rows)
    for col in stats_columns:
        if col not in stats_df.columns:
            stats_df[col] = None
    stats_df = stats_df[stats_columns]
    stats_df.to_csv(out_path, index=False)
    print(f"Wrote {len(stats_df)} fetch statistics rows to {out_path}")


# ---------------------------------------------------------------------------
# 3. Harmonization helpers
# ---------------------------------------------------------------------------

def smiles_to_inchikey(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return inchi.MolToInchiKey(mol)
    except Exception:
        return None


def fill_missing_smiles_from_inchikey(df, inchikey_col="drug_inchikey", smiles_col="drug_smiles"):
    """Best-effort backfill: RDKit can't reconstruct SMILES from InChIKey alone
    (it's a hash), so this only works if you also have the full InChI string.
    Included as a placeholder -- if a source gives InChI (not just InChIKey),
    swap MolFromInchi in here. Otherwise these rows stay without SMILES and
    should be flagged for manual lookup (e.g. via PubChem InChIKey search)."""
    df = df.copy()
    if smiles_col not in df.columns:
        df[smiles_col] = pd.NA
    if inchikey_col not in df.columns:
        return df

    missing = df[smiles_col].isna() & df[inchikey_col].notna()
    if missing.any():
        print(f"[warn] {missing.sum()} rows have InChIKey but no SMILES -- "
              f"resolve via PubChem 'compound/inchikey/{{key}}/property/CanonicalSMILES' "
              f"or drop before training.")
    return df


def value_to_pActivity(value_nm):
    """Standard pIC50/pKi/pKd-style transform: -log10(M)."""
    import math
    try:
        value_nm = float(value_nm)
        if value_nm <= 0:
            return None
        molar = value_nm * 1e-9
        return -math.log10(molar)
    except Exception:
        return None


def dedup_against_chembl(new_df, chembl_df, inchikey_col="drug_inchikey", uniprot_col="target_uniprot"):
    """Drop rows already present in the ChEMBL extract (same drug + same target)."""
    existing_pairs = set(
        zip(chembl_df[inchikey_col].fillna(""), chembl_df[uniprot_col].fillna(""))
    )
    mask = ~new_df.apply(
        lambda r: (r.get(inchikey_col, ""), r.get(uniprot_col, "")) in existing_pairs, axis=1
    )
    return new_df[mask]

def append_to_csv(df, path):
    """Append df to an existing CSV if present, otherwise create it.
    Avoids re-writing the whole file and preserves prior batch runs."""
    if df.empty:
        return
    file_exists = os.path.isfile(path)
    df.to_csv(path, mode="a", header=not file_exists, index=False)
    print(f"[append] wrote {len(df)} rows to {path} (existing file: {file_exists})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", "--targets", dest="targets", default = "data/proteins_unipro.csv",
                    help="CSV with protein identifiers (for matching) or target list")
    ap.add_argument("--chembl", default = "data/chembl_drugs_targets.csv", help="existing ChEMBL CSV with semicolon delimiters")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stats-out", default=None,
                    help="CSV file to save per-target fetch statistics")
    ap.add_argument("--skip-pubchem", action="store_true")
    ap.add_argument("--skip-bindingdb", action="store_true")
    ap.add_argument("--extract-only", action="store_true",
                    help="extract interaction tests from the local CSVs and stop")
    ap.add_argument("--max-pubchem-aids", type=int, default=20,
                    help="maximum number of PubChem assays to fetch for the taxonomy")
    ap.add_argument("--target-start", type=int, default=0,
                help="start index (inclusive) into the targets CSV for this batch")
    ap.add_argument("--target-end", type=int, default=None,
                help="end index (exclusive) into the targets CSV for this batch; default = all remaining")
    args = ap.parse_args()

    out_path = args.out or ("data/interaction_tests.csv" if args.extract_only else "data/merged_dti.csv")
    stats_out_path = args.stats_out or (os.path.splitext(out_path)[0] + "_fetch_stats.csv")

    if args.extract_only:
        extract_interaction_tests(args.targets, args.chembl, out_path)
        return

    targets = pd.read_csv(args.targets)
    total_targets = len(targets)
    end_idx = args.target_end if args.target_end is not None else total_targets
    targets = targets.iloc[args.target_start:end_idx].reset_index(drop=True)
    print(f"[batch] processing targets [{args.target_start}:{end_idx}] of {total_targets} total")
    chembl = pd.read_csv(args.chembl, sep=";")

    frames = [chembl]
    stats_rows = []

    if not args.skip_pubchem:
        pc = collect_pubchem_per_target(targets, stats_rows=stats_rows)
        if not pc.empty:
            pc["source"] = "PubChem"
            pc["activity_value_nM"] = pd.to_numeric(pc.get("raw_activity_value"), errors="coerce")
            pc["drug_inchikey"] = pc["drug_smiles"].apply(
                lambda s: smiles_to_inchikey(s) if pd.notna(s) else None
            )
            pc["pActivity"] = pc["activity_value_nM"].apply(value_to_pActivity)
            frames.append(pc)

    if not args.skip_bindingdb:
        bdb = collect_bindingdb(targets, stats_rows=stats_rows)
        if not bdb.empty:
            bdb["source"] = "BindingDB"
            bdb["drug_inchikey"] = bdb["drug_smiles"].apply(
                lambda s: smiles_to_inchikey(s) if pd.notna(s) else None
            )
            bdb["pActivity"] = bdb["activity_value_nM"].apply(value_to_pActivity)
            frames.append(bdb)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    for col in ["drug_smiles", "drug_inchikey", "activity_value_nM", "activity_label", "pActivity"]:
        if col not in merged.columns:
            merged[col] = pd.NA

    merged = fill_missing_smiles_from_inchikey(merged)

    # Final sanity: every row should carry a SMILES and an activity_label.
    # Anything missing both a quantitative and categorical signal is useless
    # for training -- flag rather than silently keep.
    n_before = len(merged)
    merged = merged[merged["drug_smiles"].notna()]
    if "activity_label" in merged.columns:
        merged["activity_label"] = merged["activity_label"].fillna("inconclusive")
    n_after = len(merged)
    if n_after < n_before:
        print(f"[info] dropped {n_before - n_after} rows lacking SMILES entirely")

    append_to_csv(merged, out_path)
    append_to_csv(pd.DataFrame(stats_rows), stats_out_path)

    print(f"Wrote {len(merged)} total rows to {out_path}")
    print(f"  ChEMBL: {len(chembl)}")
    if not args.skip_pubchem:
        print(f"  PubChem (raw, pre-dedup): {len(pc) if not pc.empty else 0}")
    if not args.skip_bindingdb:
        print(f"  BindingDB (raw, pre-dedup): {len(bdb) if not bdb.empty else 0}")


if __name__ == "__main__":
    main()