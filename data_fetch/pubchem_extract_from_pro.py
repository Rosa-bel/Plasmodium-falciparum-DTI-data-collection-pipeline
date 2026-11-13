#!/usr/bin/env python3
"""
pubchem_pfalciparum.py
=======================

Retrieves, via the PubChem PUG REST API, all proteins associated with the
organism *Plasmodium falciparum*, and for each protein, all compounds that
have been tested against it in PubChem BioAssay -- both active and
inactive -- along with the activity outcome field.

Pipeline
--------
1. Resolve the NCBI Taxonomy ID for "Plasmodium falciparum" (taxid 5833).
2. Use the `taxonomy/taxid/<id>/proteins` PUG REST operation to list every
   PubChem protein (accession) tied to that organism.
3. For each protein accession, fetch its summary (name, gene ID(s)) via
   `protein/accession/<acc>/summary`.
4. For each protein accession, fetch the list of BioAssay AIDs that tested
   it via `protein/accession/<acc>/aids`.
5. For each AID, fetch the per-record assay CSV (`assay/aid/<aid>/CSV`),
   which gives one row per tested substance (SID/CID) with its
   PUBCHEM_ACTIVITY_OUTCOME plus whatever extra readout columns that
   particular assay deposited (Standard Type/Relation/Value/Units,
   Inhibition replicate columns, etc. -- these vary per assay).
6. Aggregate everything into a single JSON checkpoint file, flatten a LEAN
   core CSV of (protein, compound, outcome) rows, and ALSO write every
   single raw column captured per compound into a SEPARATE "raw" CSV so
   the lean CSV doesn't get overloaded with assay-specific columns. No
   interpretation/classification of activity is done here -- this is pure
   extraction.

Notes / caveats
----------------
- PubChem BioAssay data for some targets can be very large (assays with
  tens of thousands of tested compounds). The per-record CSV endpoint
  returns everything in one shot per AID, so very large AIDs may take a
  while to download. A `MAX_COMPOUNDS_PER_AID` safety cap is provided (set
  to None to disable).
- PubChem enforces rate limits (no more than ~5 requests/second, and asks
  for politeness). This script sleeps between requests and retries on
  HTTP 429/503 with exponential backoff.
- Different assays deposit different "extra" result columns (e.g. some
  have "Standard Type/Relation/Value/Units", others have raw
  "Inhibition 1".."Inhibition 23" replicate columns, etc.). Rather than
  forcing every assay into one fixed schema, every non-core column found
  for a given row is captured into a single JSON-encoded `extra_fields`
  string in the raw CSV -- safe to append across assays with differing
  columns, and nothing is silently dropped.
- Results are checkpointed to disk (JSON) as we go, so the script can be
  interrupted and re-run without losing previously-fetched proteins. The
  raw and lean CSVs are also written incrementally (appended) as each
  protein finishes, not just once at the very end.

Usage
-----
    pip install requests pandas --break-system-packages
    python pubchem_pfalciparum.py
"""

import csv
import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

ORGANISM_NAME = "Plasmodium falciparum"
ORGANISM_TAXID = 5833  # well-known NCBI taxid for P. falciparum

OUT_DIR = Path("./data/pubchem_pfalciparum")
OUT_DIR.mkdir(exist_ok=True)

CHECKPOINT_FILE = OUT_DIR / "pfalciparum_proteins_checkpoint.json"
FINAL_CSV = OUT_DIR / "pfalciparum_protein_compound_activity_pubchem.csv"
RAW_COMPOUNDS_CSV = OUT_DIR / "pfalciparum_compounds_raw.csv"

SLEEP_BETWEEN_REQUESTS = 0.25   # seconds, politeness delay
MAX_RETRIES = 5
MAX_COMPOUNDS_PER_AID = 5000    # e.g. set to 5000 to cap huge assays, or None for no cap

# Core columns from the per-record CSV that always get their own column in
# BOTH the lean and raw CSVs. Everything else found in a given assay's CSV
# (Standard Type/Relation/Value/Units, Inhibition replicate columns,
# PUBCHEM_ASSAYDATA_COMMENT, PUBCHEM_ACTIVITY_SCORE/URL, etc.) is preserved
# in the raw CSV's `extra_fields` JSON blob instead of forcing a fixed schema.
CORE_COLUMNS = {
    "PUBCHEM_RESULT_TAG", "PUBCHEM_SID", "PUBCHEM_CID",
    "PUBCHEM_EXT_DATASOURCE_SMILES", "PUBCHEM_ACTIVITY_OUTCOME",
}


# --------------------------------------------------------------------------
# Low-level request helper with retry/backoff
# --------------------------------------------------------------------------
def pug_get(path, accept="JSON", params=None):
    """
    GET a PUG REST endpoint, e.g. path='taxonomy/taxid/5833/proteins'.
    Returns parsed JSON (dict) or None on persistent failure / 404 (no data).
    """
    url = f"{BASE}/{path}/{accept}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=60)
        except requests.RequestException as exc:
            print(f"  [warn] network error on {url}: {exc} (attempt {attempt})")
            time.sleep(2 * attempt)
            continue

        if resp.status_code == 200:
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            if accept == "JSON":
                try:
                    return resp.json()
                except ValueError:
                    print(f"  [warn] non-JSON response from {url}")
                    return None
            return resp.text

        if resp.status_code == 404:
            # No data for this query -- not an error, just empty.
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            return None

        if resp.status_code in (429, 503):
            wait = 2 ** attempt
            print(f"  [warn] HTTP {resp.status_code} on {url}, backing off {wait}s "
                  f"(attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait)
            continue

        print(f"  [warn] HTTP {resp.status_code} on {url}: {resp.text[:200]}")
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        return None

    print(f"  [error] giving up on {url} after {MAX_RETRIES} attempts")
    return None


def pug_get_csv_text(path):
    """
    GET a PUG REST endpoint that returns CSV (not JSON), e.g.
    path='assay/aid/1000/CSV'. Returns raw response text, or None on
    persistent failure / 404 (no data).
    """
    url = f"{BASE}/{path}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=60)
        except requests.RequestException as exc:
            print(f"  [warn] network error on {url}: {exc} (attempt {attempt})")
            time.sleep(2 * attempt)
            continue

        if resp.status_code == 200:
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            return resp.text

        if resp.status_code == 404:
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            return None

        if resp.status_code in (429, 503):
            wait = 2 ** attempt
            print(f"  [warn] HTTP {resp.status_code} on {url}, backing off {wait}s "
                  f"(attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait)
            continue

        print(f"  [warn] HTTP {resp.status_code} on {url}: {resp.text[:200]}")
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        return None

    print(f"  [error] giving up on {url} after {MAX_RETRIES} attempts")
    return None


# --------------------------------------------------------------------------
# Step 3: protein summary (name, gene id)
# --------------------------------------------------------------------------
def get_protein_summary(accession):
    """
    Returns dict with keys: accession, name, gene_id, organism (best-effort
    parse; PubChem's summary JSON schema can vary slightly by record).
    """
    data = pug_get(f"protein/accession/{accession}/summary")
    result = {
        "accession": accession,
        "name": None,
        "gene_id": None,
        "organism": None,
    }
    if not data:
        return result

    try:
        info = data["ProteinSummaries"]["ProteinSummary"][0]
    except (KeyError, IndexError, TypeError):
        return result

    result["name"] = info.get("Name") or info.get("Title")
    result["organism"] = info.get("TaxName") or info.get("Organism")

    # Gene ID may show up as 'GeneID', a list under 'Gene', or similar.
    gene_id = info.get("GeneID")
    if gene_id is None:
        genes = info.get("Gene") or info.get("Genes")
        if isinstance(genes, list) and genes:
            gene_id = genes[0].get("GeneID") if isinstance(genes[0], dict) else genes[0]
        elif isinstance(genes, dict):
            gene_id = genes.get("GeneID")
    result["gene_id"] = gene_id

    return result


# --------------------------------------------------------------------------
# Step 4: AIDs (bioassays) that tested a given protein
# --------------------------------------------------------------------------
def get_aids_for_protein(accession):
    data = pug_get(f"protein/accession/{accession}/aids")
    if not data:
        return []
    try:
        print("Fetching aids for protein : ", accession)
        id_list = data["InformationList"]["Information"][0]["AID"]
        return id_list if isinstance(id_list, list) else [id_list]
    except (KeyError, IndexError, TypeError):
        return []


# --------------------------------------------------------------------------
# Step 5: compound test results (active/inactive) for a given AID
#
# Uses the per-record assay CSV endpoint (assay/aid/{aid}/CSV) instead of
# the 'concise' JSON endpoint. This is a pure EXTRACTION step: every column
# present in the CSV is captured (core fields get their own dict key; every
# other column -- Standard Type/Relation/Value/Units, Inhibition replicate
# columns, comments, score, url, etc. -- is kept verbatim, unparsed, in
# 'extra_fields'). No active/inactive interpretation happens here.
# --------------------------------------------------------------------------
def get_compounds_for_aid(aid):
    """
    Returns a list of dicts, one per tested substance row in the assay's
    per-record CSV:
        {
          "aid": aid,
          "sid": ...,
          "cid": ...,
          "smiles": ...,
          "activity_outcome": ...,   # raw PUBCHEM_ACTIVITY_OUTCOME, uninterpreted
          "extra_fields": {...}      # every other column verbatim, e.g.
                                      # Standard Type/Relation/Value/Units,
                                      # PUBCHEM_ACTIVITY_SCORE, PUBCHEM_ACTIVITY_URL,
                                      # PUBCHEM_ASSAYDATA_COMMENT, replicate columns...
        }
    Skips the metadata rows (RESULT_TYPE / RESULT_DESCR / RESULT_UNIT /
    RESULT_ATTR_CONC_MICROMOL...) by checking PUBCHEM_RESULT_TAG is numeric.
    """
    text = pug_get_csv_text(f"assay/aid/{aid}/CSV")
    if not text or "PUBCHEM_SID" not in text:
        return []

    print(f"Fetching compounds for AID: {aid}")

    results = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        tag = (row.get("PUBCHEM_RESULT_TAG") or "").strip()
        if not tag.isdigit():
            # metadata rows: RESULT_TYPE, RESULT_DESCR, RESULT_UNIT,
            # RESULT_ATTR_CONC_MICROMOL, etc. -- not real data rows.
            continue

        sid = (row.get("PUBCHEM_SID") or "").strip()
        if not sid:
            continue

        extra_fields = {
            k: v for k, v in row.items()
            if k not in CORE_COLUMNS and v not in (None, "")
        }

        results.append({
            "aid": aid,
            "sid": sid,
            "cid": (row.get("PUBCHEM_CID") or "").strip(),
            "smiles": (row.get("PUBCHEM_EXT_DATASOURCE_SMILES") or "").strip(),
            "activity_outcome": (row.get("PUBCHEM_ACTIVITY_OUTCOME") or "").strip(),
            "extra_fields": extra_fields,
        })

        if MAX_COMPOUNDS_PER_AID and len(results) >= MAX_COMPOUNDS_PER_AID:
            break

    print(f"  AID {aid}: {len(results)} tested compounds extracted")
    return results


# --------------------------------------------------------------------------
# Checkpointing helpers
# --------------------------------------------------------------------------
def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(proteins_dict):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(proteins_dict, f, indent=2)


# --------------------------------------------------------------------------
# Incremental CSV writers (lean + raw)
# --------------------------------------------------------------------------
def append_lean_rows(rows, summary, acc):
    """Lean CSV: one row per compound, CORE fields only (no extra columns)."""
    file_exists = FINAL_CSV.exists()
    with open(FINAL_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "protein_accession", "protein_name", "gene_id", "organism",
                "aid", "cid", "sid", "smiles", "activity_outcome",
            ])
        for comp in rows:
            writer.writerow([
                acc, summary["name"], summary["gene_id"],
                summary["organism"] or ORGANISM_NAME,
                comp["aid"], comp["cid"], comp["sid"], comp["smiles"],
                comp["activity_outcome"],
            ])


def append_raw_rows(rows, acc):
    """
    Raw CSV: one row per compound, core fields + a single JSON-encoded
    'extra_fields' column capturing every other assay-specific column
    verbatim. Kept SEPARATE from the lean CSV so differing per-assay
    schemas never collide with / bloat the main file.
    """
    file_exists = RAW_COMPOUNDS_CSV.exists()
    with open(RAW_COMPOUNDS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "protein_accession", "aid", "sid", "cid", "smiles",
                "activity_outcome", "extra_fields_json",
            ])
        for comp in rows:
            writer.writerow([
                acc, comp["aid"], comp["sid"], comp["cid"], comp["smiles"],
                comp["activity_outcome"], json.dumps(comp["extra_fields"], ensure_ascii=False),
            ])


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def main():
    proteins_dict = load_checkpoint()
    already_done = set(proteins_dict.keys())
    if already_done:
        print(f"Resuming from checkpoint: {len(already_done)} proteins already processed.")

    accessions = pd.read_csv(
        "data/uniprotkb_organism_name_Plasmodium_falc_2026_06_19.tsv", sep="\t"
    )["Entry"].tolist()
    remaining = [a for a in accessions if a not in already_done]
    print(f"{len(remaining)} proteins left to process "
          f"(out of {len(accessions)} total).")

    for i, acc in enumerate(remaining, 1):
        print(f"[{i}/{len(remaining)}] Protein {acc}")

        summary = get_protein_summary(acc)
        aids = get_aids_for_protein(acc)
        print(f"    name={summary['name']!r}  gene_id={summary['gene_id']!r}  "
              f"#assays={len(aids)}")

        compounds = []
        for j, aid in enumerate(aids, 1):
            comps = get_compounds_for_aid(aid)
            print(f"      AID {aid}: {len(comps)} tested compounds "
                  f"({j}/{len(aids)})")
            compounds.extend(comps)

        proteins_dict[acc] = {
            "accession": acc,
            "name": summary["name"],
            "gene_id": summary["gene_id"],
            "organism": summary["organism"] or ORGANISM_NAME,
            "num_assays": len(aids),
            "aids": aids,
            "num_compounds": len(compounds),  # counts only -- raw per-compound
                                               # data lives in the raw CSV, not
                                               # duplicated into the checkpoint.
        }

        # write incrementally as we go, not just once at the very end
        append_lean_rows(compounds, summary, acc)
        append_raw_rows(compounds, acc)

        # Checkpoint after every protein so we never lose progress.
        save_checkpoint(proteins_dict)

    print(f"Lean CSV written to {FINAL_CSV}")
    print(f"Raw (uninterpreted, all-fields) CSV written to {RAW_COMPOUNDS_CSV}")


if __name__ == "__main__":
    main()