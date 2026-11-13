#!/usr/bin/env python3
"""
pfalciparum_bioassay_scraper.py  (v4)

Searches NCBI PubChem BioAssay (pcassay) for all assays mentioning
"Plasmodium falciparum", then for each assay (batched):
  - checks for a real protein-level target via the PUG-REST `targets` endpoint:
        /assay/aid/{aid1,aid2,...}/targets/ProteinGI,ProteinName,GeneID,GeneSymbol/JSON
    -> if PubChem returns "No targets found for the given AID(s)" for an AID,
       that assay is skipped (no molecular target).
  - if a target IS found, fetches the assay DESCRIPTION
        /assay/aid/{aid}/description/JSON
    and looks in its "Comment" lines for a "Target ChEMBL ID: CHEMBLxxxxx"
    entry. That ChEMBL target ID is then queried against the ChEMBL REST API
        https://www.ebi.ac.uk/chembl/api/data/target/{chembl_id}.json
    to read the target's `organism` / `tax_id`. The assay is KEPT only if the
    organism resolves to Plasmodium falciparum (organism string contains
    "Plasmodium falciparum", or tax_id == 5833).
  - if no ChEMBL ID is present in the comments at all, the assay is skipped
    (we have no organism to verify against -- conservative default).
  - if a target IS found and confirmed as P. falciparum, retrieves ALL tested
    substances/compounds for that assay via the per-record CSV endpoint:
        /assay/aid/{aid}/CSV
    This returns one row per tested SID with its CID, SMILES, and
    PUBCHEM_ACTIVITY_OUTCOME (Active / Inactive / Inconclusive / Unspecified /
    Probe / Discrepant) -- the authoritative outcome field, instead of the
    coarse sids_type=active/inactive filter (which misses anything not coded
    exactly outcome=1/2, e.g. ChEMBL-derived assays that are often left
    "Unspecified" even when a Standard Text Value of "Active" is present).
    SIDs are bucketed into active_compounds / inactive_compounds based on
    this PUBCHEM_ACTIVITY_OUTCOME column.

Output (two CSV files):

  1) --out (main results), columns:
       assay_id, protein_target, gene_symbol, gi, gene_id, chembl_target_id,
       chembl_organism, active_compounds, inactive_compounds
     (active/inactive_compounds are ';'-joined PubChem SIDs)

  2) --compounds-out, columns:
       pubchem_cid, pubchem_sid, smiles
     One row per unique (cid, sid) pair seen across ALL processed assays,
     for every tested compound that has a resolvable SMILES (regardless of
     outcome -- active, inactive, inconclusive, etc. all get a SMILES row
     here; only the main CSV buckets by outcome).

Usage:
    python pfalciparum_bioassay_scraper.py --out pf_bioassays.csv --compounds-out pf_compounds.csv
    python pfalciparum_bioassay_scraper.py --out pf_bioassays.csv --compounds-out pf_compounds.csv --max-assays 500
    python pfalciparum_bioassay_scraper.py --out pf_bioassays.csv --compounds-out pf_compounds.csv --resume

Notes:
  - The /targets/ endpoint is queried in batches (default 100 AIDs per call).
  - The ChEMBL organism lookup is cached in-memory by chembl_target_id.
  - Writes incrementally; safe to kill/resume with --resume (resume dedupes
    against both output files).
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import requests

ENTREZ_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUG_TARGETS = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aids}/targets/ProteinGI,ProteinName,GeneID,GeneSymbol/JSON"
PUG_DESCRIPTION = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/description/JSON"
PUG_ASSAY_CSV = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV"
CHEMBL_TARGET = "https://www.ebi.ac.uk/chembl/api/data/target/{chembl_id}.json"

REQUEST_SLEEP = 0.35        # ~3 req/sec, safe under NCBI's no-key rate limit
RETRY_SLEEP = 5
MAX_RETRIES = 4
ESEARCH_PAGE_SIZE = 500
TARGETS_BATCH_SIZE = 100    # AIDs per /targets/ call

PFALCIPARUM_TAX_ID = "5833"
CHEMBL_ID_RE = re.compile(r"Target ChEMBL ID:\s*(CHEMBL\d+)", re.IGNORECASE)

ACTIVE_OUTCOMES = {"active", "probe"}
INACTIVE_OUTCOMES = {"inactive"}
# everything else (Inconclusive, Unspecified, Discrepant) is tracked
# but not bucketed into active/inactive_compounds, unless rescued by the
# Standard Text Value fallback in bucket_outcomes().

# in-memory cache: chembl_target_id -> (organism_str, tax_id_str)
_chembl_cache = {}

g
def log(msg):
    print(msg, file=sys.stderr, flush=True)


def safe_get(url, params=None, max_retries=MAX_RETRIES):
    """GET with retry/backoff. Returns requests.Response or None on persistent failure."""
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return r  # often means "no data" -- caller decides
            if r.status_code == 429:
                log(f"  [429 rate-limited] backing off... (attempt {attempt})")
                time.sleep(RETRY_SLEEP * attempt)
                continue
            log(f"  [HTTP {r.status_code}] {url} (attempt {attempt})")
            time.sleep(RETRY_SLEEP)
        except requests.RequestException as e:
            log(f"  [request error] {e} (attempt {attempt})")
            time.sleep(RETRY_SLEEP)
    return None


def search_all_pfalciparum_aids(retmax_total=None):
    """
    Search pcassay for all assays mentioning Plasmodium falciparum.
    Paginates via retstart. Returns a list of AID strings.
    """
    term = '"Plasmodium falciparum"[All Fields]'
    all_ids = []
    retstart = 0

    while True:
        params = {
            "db": "pcassay",
            "term": term,
            "retstart": retstart,
            "retmax": ESEARCH_PAGE_SIZE,
            "retmode": "json",
        }
        r = safe_get(ENTREZ_ESEARCH, params=params)
        if r is None:
            log(f"  esearch failed at retstart={retstart}, stopping pagination.")
            break

        data = r.json()
        result = data.get("esearchresult", {})
        ids = result.get("idlist", [])
        count = int(result.get("count", 0))

        if retstart == 0:
            log(f"Total assays found mentioning 'Plasmodium falciparum': {count}")

        if not ids:
            break

        all_ids.extend(ids)
        retstart += len(ids)
        log(f"  fetched {len(all_ids)}/{count} assay IDs so far...")

        if retmax_total and len(all_ids) >= retmax_total:
            all_ids = all_ids[:retmax_total]
            break
        if retstart >= count:
            break

        time.sleep(REQUEST_SLEEP)

    return all_ids


def fetch_targets_batch(aid_batch):
    """
    Query the /targets/ endpoint for a batch of AIDs.
    Returns a dict: {aid_str: {"ProteinName": [...], "GeneSymbol": [...], "GeneID": [...], "ProteinGI": [...]}}
    AIDs with no targets are NOT included in the returned dict.
    """
    aids_str = ",".join(str(a) for a in aid_batch)
    url = PUG_TARGETS.format(aids=aids_str)
    r = safe_get(url)

    result = {}
    if r is None:
        log(f"  [targets batch failed] {aids_str[:60]}...")
        return result

    if r.status_code == 404:
        return result

    try:
        data = r.json()
    except json.JSONDecodeError:
        log(f"  [targets batch] could not parse JSON for {aids_str[:60]}...")
        return result

    if "Fault" in data or (isinstance(data, dict) and "Message" in data):
        return result

    info_list = data.get("InformationList", {}).get("Information", [])
    for entry in info_list:
        aid = entry.get("AID")
        if aid is None:
            continue
        protein_names = entry.get("ProteinName", [])
        gene_symbols = entry.get("GeneSymbol", [])
        gene_ids = entry.get("GeneID", [])
        protein_gis = entry.get("ProteinGI", [])
        if not (protein_names or gene_symbols or gene_ids or protein_gis):
            continue
        result[str(aid)] = {
            "ProteinName": protein_names,
            "GeneSymbol": gene_symbols,
            "GeneID": gene_ids,
            "ProteinGI": protein_gis,
        }
    return result


def fetch_assay_comments(aid):
    """
    Fetch the assay description for a single AID and return its list of
    "Comment" strings (where the "Target ChEMBL ID: CHEMBLxxxx" line lives).
    Returns [] on failure / no description.
    """
    url = PUG_DESCRIPTION.format(aid=aid)
    r = safe_get(url)
    if r is None or r.status_code != 200:
        return []
    try:
        data = r.json()
    except json.JSONDecodeError:
        return []

    try:
        assays = data.get("PC_AssayContainer", [])
        comments = []
        for assay in assays:
            descr = assay.get("assay", {}).get("descr", {})
            comments.extend(descr.get("comment", []))
        return comments
    except (AttributeError, TypeError):
        return []


def extract_chembl_target_id(comments):
    for line in comments:
        m = CHEMBL_ID_RE.search(line)
        if m:
            return m.group(1).upper()
    return None


def get_chembl_organism(chembl_target_id):
    """
    Query the ChEMBL REST API for a target's organism / tax_id.
    Returns (organism_str, tax_id_str) or (None, None) on failure.
    Cached in-memory by chembl_target_id.
    """
    if chembl_target_id in _chembl_cache:
        return _chembl_cache[chembl_target_id]

    url = CHEMBL_TARGET.format(chembl_id=chembl_target_id)
    r = safe_get(url)
    if r is None or r.status_code != 200:
        _chembl_cache[chembl_target_id] = (None, None)
        return (None, None)

    try:
        data = r.json()
        organism = data.get("organism")
        tax_id = data.get("tax_id")
        tax_id = str(tax_id) if tax_id is not None else None
        result = (organism, tax_id)
    except (json.JSONDecodeError, KeyError):
        result = (None, None)

    _chembl_cache[chembl_target_id] = result
    return result


def is_pfalciparum_organism(organism, tax_id):
    if tax_id == PFALCIPARUM_TAX_ID:
        return True
    if organism and "plasmodium falciparum" in organism.lower():
        return True
    return False


def fetch_assay_csv_rows(aid):
    """
    Download the per-record assay CSV (/assay/aid/{aid}/CSV) and parse it
    into a list of dicts, one per tested substance, with keys:
        sid, cid, smiles, outcome (raw PUBCHEM_ACTIVITY_OUTCOME string)
    Skips the metadata header rows (RESULT_TYPE / RESULT_UNIT /
    RESULT_ATTR_CONC_MICROMOL) by checking that PUBCHEM_RESULT_TAG is numeric.
    Returns [] on failure or if the assay has no per-record data.
    """
    url = PUG_ASSAY_CSV.format(aid=aid)
    r = safe_get(url)
    if r is None or r.status_code != 200:
        return []

    text = r.text
    if not text or "PUBCHEM_SID" not in text:
        return []

    rows = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        tag = (row.get("PUBCHEM_RESULT_TAG") or "").strip()
        if not tag.isdigit():
            # metadata rows: RESULT_TYPE, RESULT_UNIT, RESULT_ATTR_CONC_MICROMOL...
            continue

        sid = (row.get("PUBCHEM_SID") or "").strip()
        if not sid:
            continue

        cid = (row.get("PUBCHEM_CID") or "").strip()
        smiles = (row.get("PUBCHEM_EXT_DATASOURCE_SMILES") or "").strip()
        outcome = (row.get("PUBCHEM_ACTIVITY_OUTCOME") or "").strip()
        std_text_value = (row.get("Standard Text Value") or "").strip()

        rows.append({
            "sid": sid,
            "cid": cid,
            "smiles": smiles,
            "outcome": outcome,
            "std_text_value": std_text_value,
        })

    return rows


def bucket_outcomes(csv_rows):
    """
    Given parsed CSV rows (from fetch_assay_csv_rows), bucket SIDs by
    PUBCHEM_ACTIVITY_OUTCOME, falling back to the "Standard Text Value"
    column (e.g. "Active"/"Inactive") whenever PubChem's own outcome field
    is "Unspecified" (or blank) -- this covers ChEMBL-derived assays that
    only carry a categorical text result rather than a numeric/outcome-coded
    one. Returns (active_sids, inactive_sids, other_count).
    """
    active_sids = []
    inactive_sids = []
    other_count = 0

    for row in csv_rows:
        outcome_lower = row["outcome"].lower()

        if outcome_lower in ACTIVE_OUTCOMES:
            active_sids.append(row["sid"])
            continue
        if outcome_lower in INACTIVE_OUTCOMES:
            inactive_sids.append(row["sid"])
            continue

        if outcome_lower in ("unspecified", ""):
            std_text_lower = row.get("std_text_value", "").lower()
            if std_text_lower in ACTIVE_OUTCOMES:
                active_sids.append(row["sid"])
                continue
            if std_text_lower in INACTIVE_OUTCOMES:
                inactive_sids.append(row["sid"])
                continue

        other_count += 1

    return active_sids, inactive_sids, other_count


def load_done_aids(csv_path):
    done = set()
    if csv_path and os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(str(row["assay_id"]))
    return done


def load_known_compound_keys(compounds_csv_path):
    """Returns a set of (cid, sid) tuples already written to the compounds CSV."""
    known = set()
    if compounds_csv_path and os.path.exists(compounds_csv_path):
        with open(compounds_csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                known.add((row.get("pubchem_cid", ""), row.get("pubchem_sid", "")))
    return known


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="pf_bioassays.csv", help="Main output CSV path")
    parser.add_argument("--compounds-out", default="pf_compounds.csv",
                         help="Output CSV path for cid/sid/smiles of all tested compounds")
    parser.add_argument("--max-assays", type=int, default=None,
                         help="Limit total number of assays searched (for testing)")
    parser.add_argument("--resume", action="store_true",
                         help="Resume: skip AIDs already present in --out")
    parser.add_argument("--targets-batch-size", type=int, default=TARGETS_BATCH_SIZE,
                         help=f"AIDs per /targets/ batch call (default {TARGETS_BATCH_SIZE})")
    parser.add_argument("--max-compounds-per-row", type=int, default=None,
                         help="Optional cap on number of SIDs stored per active/inactive "
                              "field in the main CSV (does not affect the compounds CSV).")
    parser.add_argument("--no-chembl-filter", action="store_true",
                         help="Disable the ChEMBL-organism taxonomy check.")
    args = parser.parse_args()

    done_aids = load_done_aids(args.out) if args.resume else set()
    known_compound_keys = load_known_compound_keys(args.compounds_out) if args.resume else set()
    if done_aids:
        log(f"Resuming: {len(done_aids)} assays already in {args.out}, will skip them.")

    log("Searching PubChem BioAssay (pcassay) for 'Plasmodium falciparum'...")
    all_aids = search_all_pfalciparum_aids(retmax_total=args.max_assays)
    aids_to_process = [a for a in all_aids if a not in done_aids]
    log(f"Total assays found: {len(all_aids)} | to process this run: {len(aids_to_process)}")

    main_file_exists = os.path.exists(args.out)
    main_write_header = not (args.resume and main_file_exists)
    main_mode = "a" if (args.resume and main_file_exists) else "w"

    comp_file_exists = os.path.exists(args.compounds_out)
    comp_write_header = not (args.resume and comp_file_exists)
    comp_mode = "a" if (args.resume and comp_file_exists) else "w"

    n_kept = 0
    n_skipped_no_target = 0
    n_skipped_no_chembl_id = 0
    n_skipped_wrong_organism = 0
    n_compound_rows_written = 0

    with open(args.out, main_mode, newline="", encoding="utf-8") as f_main, \
         open(args.compounds_out, comp_mode, newline="", encoding="utf-8") as f_comp:

        main_writer = csv.writer(f_main)
        comp_writer = csv.writer(f_comp)

        if main_write_header:
            main_writer.writerow([
                "assay_id", "protein_target", "gene_symbol", "gi", "gene_id",
                "chembl_target_id", "chembl_organism",
                "active_compounds", "inactive_compounds"
            ])
            f_main.flush()

        if comp_write_header:
            comp_writer.writerow(["pubchem_cid", "pubchem_sid", "smiles"])
            f_comp.flush()

        for batch_num, batch in enumerate(chunked(aids_to_process, args.targets_batch_size), 1):
            log(f"[batch {batch_num}] checking targets for {len(batch)} AIDs "
                f"({batch[0]}..{batch[-1]}) ...")

            targets_map = fetch_targets_batch(batch)
            time.sleep(REQUEST_SLEEP)

            for aid in batch:
                aid_str = str(aid)
                target_info = targets_map.get(aid_str)

                if not target_info:
                    log(f"  AID {aid_str}: no targets found, skipping.")
                    n_skipped_no_target += 1
                    continue

                chembl_target_id = None
                organism = None

                if not args.no_chembl_filter:
                    comments = fetch_assay_comments(aid_str)
                    time.sleep(REQUEST_SLEEP)
                    chembl_target_id = extract_chembl_target_id(comments)

                    if not chembl_target_id:
                        log(f"  AID {aid_str}: no 'Target ChEMBL ID' in comments, skipping.")
                        n_skipped_no_chembl_id += 1
                        continue

                    organism, tax_id = get_chembl_organism(chembl_target_id)
                    time.sleep(REQUEST_SLEEP)

                    if not is_pfalciparum_organism(organism, tax_id):
                        log(f"  AID {aid_str}: ChEMBL target {chembl_target_id} organism="
                            f"'{organism}' (tax_id={tax_id}) is not P. falciparum, skipping.")
                        n_skipped_wrong_organism += 1
                        continue

                    log(f"  AID {aid_str}: ChEMBL target {chembl_target_id} confirmed "
                        f"organism='{organism}' -- proceeding.")

                log(f"  AID {aid_str}: target found -> "
                    f"{target_info['ProteinName'][:1]} / gene={target_info['GeneSymbol'][:1]} "
                    f"-- fetching assay CSV for tested compounds...")

                csv_rows = fetch_assay_csv_rows(aid_str)
                time.sleep(REQUEST_SLEEP)

                active_sids, inactive_sids, other_count = bucket_outcomes(csv_rows)

                if args.max_compounds_per_row:
                    active_sids = active_sids[: args.max_compounds_per_row]
                    inactive_sids = inactive_sids[: args.max_compounds_per_row]

                protein_target_str = "; ".join(str(x) for x in target_info["ProteinName"])
                gene_symbol_str = "; ".join(str(x) for x in target_info["GeneSymbol"])
                gi_str = "; ".join(str(x) for x in target_info.get("ProteinGI", []))
                gene_id_str = "; ".join(str(x) for x in target_info["GeneID"])
                active_str = "; ".join(active_sids)
                inactive_str = "; ".join(inactive_sids)

                main_writer.writerow([
                    aid_str, protein_target_str, gene_symbol_str, gi_str, gene_id_str,
                    chembl_target_id or "", organism or "",
                    active_str, inactive_str
                ])
                f_main.flush()
                n_kept += 1

                # write every tested compound with a resolvable SMILES to the
                # compounds CSV, deduped by (cid, sid), regardless of outcome
                new_compound_rows = 0
                for row in csv_rows:
                    if not row["smiles"]:
                        continue
                    key = (row["cid"], row["sid"])
                    if key in known_compound_keys:
                        continue
                    known_compound_keys.add(key)
                    comp_writer.writerow([row["cid"], row["sid"], row["smiles"]])
                    new_compound_rows += 1
                if new_compound_rows:
                    f_comp.flush()
                n_compound_rows_written += new_compound_rows

                log(f"    -> kept. n_active={len(active_sids)} n_inactive={len(inactive_sids)} "
                    f"n_other_outcome={other_count} n_new_compound_rows={new_compound_rows}")

    log("")
    log("=== Done ===")
    log(f"Assays kept (confirmed P. falciparum target):  {n_kept}")
    log(f"Assays skipped (no protein target):            {n_skipped_no_target}")
    log(f"Assays skipped (no ChEMBL target ID in desc.): {n_skipped_no_chembl_id}")
    log(f"Assays skipped (wrong organism / homologous):  {n_skipped_wrong_organism}")
    log(f"Compound (cid/sid/smiles) rows written:        {n_compound_rows_written}")
    log(f"Main output written to:      {args.out}")
    log(f"Compounds output written to: {args.compounds_out}")


if __name__ == "__main__":
    main()