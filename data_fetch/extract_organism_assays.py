#!/usr/bin/env python3
"""Extract assays (Assay Type != 'B' and Tax ID == 5833) and fetch activities from ChEMBL.

Saves a CSV to `data/organism_assays.csv` with activity and molecule info.
"""
import csv
import time
import argparse

import requests

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


def read_and_filter_chembl(csv_path):
    assays = []
    with open(csv_path, newline='') as fh:
        reader = csv.DictReader(fh, delimiter=';', quotechar='"')
        for row in reader:
            assay_type = row.get('Assay Type', '').strip()
            tax_id = row.get('Tax ID', '').strip()
            assay_id = row.get('Assay ChEMBL ID', '').strip()
            if assay_id == '':
                continue
            if assay_type != 'B' and tax_id == '5833':
                assays.append(row)
    return assays


def fetch_activities_for_assay(assay_chembl_id, session=None, sleep=0.1):
    """Fetch all activities for a given assay_chembl_id from ChEMBL.

    Returns a list of activity dicts from the API.
    """
    session = session or requests.Session()
    activities = []
    url = f"{CHEMBL_BASE}/activity.json"
    print("fetching activities for assay", assay_chembl_id)
    params = {"assay_chembl_id": assay_chembl_id, "limit": 1000, "offset": 0}
    while True:
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"ChEMBL API error {resp.status_code}: {resp.text}")
        data = resp.json()
        # API returns results under 'activities' or 'activities' key; handle common variants
        batch = data.get('activities') or data.get('activities') or data.get('results') or data.get('docs') or []
        # fallback: check top-level lists
        if not isinstance(batch, list):
            # try to find list-valued entry
            for v in data.values():
                if isinstance(v, list):
                    batch = v
                    break
        if not batch:
            break
        activities.extend(batch)
        # pagination: try offset + limit
        if 'page_meta' in data:
            meta = data['page_meta']
            if meta.get('next'):
                params['offset'] = params.get('offset', 0) + params.get('limit', 1000)
                time.sleep(sleep)
                continue
            else:
                break
        # fallback: stop when fewer than limit returned
        if len(batch) < params['limit']:
            break
        params['offset'] = params.get('offset', 0) + params['limit']
        time.sleep(sleep)
    return activities


def fetch_molecule_details(mol_id, session=None):
    session = session or requests.Session()
    url = f"{CHEMBL_BASE}/molecule/{mol_id}.json"
    resp = session.get(url, timeout=30)
    if resp.status_code != 200:
        return {}
    data = resp.json()
    pref_name = data.get('pref_name') or data.get('molecule_pref_name') or ''
    mol_struct = data.get('molecule_structures') or {}
    smiles = mol_struct.get('canonical_smiles') if isinstance(mol_struct, dict) else ''
    return {'pref_name': pref_name, 'canonical_smiles': smiles}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/chembl.csv', help='Path to chembl.csv')
    parser.add_argument('--output', default='data/organism_assays.csv', help='Output CSV path')
    parser.add_argument('--pause', type=float, default=0.12, help='Pause between API requests (s)')
    args = parser.parse_args()

    assays = read_and_filter_chembl(args.input)
    if not assays:
        print('No assays matched filter')
        return

    assay_ids = sorted({row['Assay ChEMBL ID'] for row in assays})
    session = requests.Session()

    all_activity_rows = []
    seen_molecules = set()
    molecule_cache = {}

    print(f'Fetching activities for {len(assay_ids)} assays...')
    for aid in assay_ids:
        try:
            activities = fetch_activities_for_assay(aid, session=session, sleep=args.pause)
        except Exception as e:
            print(f'Failed fetching activities for {aid}: {e}')
            continue
        for act in activities:
            activity_chembl_id = act.get('activity_chembl_id') or act.get('activity_id')
            molecule_chembl_id = act.get('molecule_chembl_id') or act.get('molregno')
            row = {
                'assay_chembl_id': aid,
                'activity_chembl_id': activity_chembl_id,
                'molecule_chembl_id': molecule_chembl_id,
                'pchembl_value': act.get('pchembl_value'),
                'potential_duplicate': act.get('potential_duplicate'),
                'qudt_units': act.get('qudt_units'),
                'record_id': act.get('record_id') or act.get('activity_id') or act.get('activity_chembl_id'),
                'relation': act.get('relation'),
                'src_id': act.get('src_id') or act.get('src'),
                'standard_flag': act.get('standard_flag'),
                'standard_relation': act.get('standard_relation'),
                'standard_text_value': act.get('standard_text_value'),
                'standard_type': act.get('standard_type'),
                'standard_units': act.get('standard_units'),
                'standard_upper_value': act.get('standard_upper_value'),
                'standard_value': act.get('standard_value'),
                'document_chembl_id': act.get('document_chembl_id'),
            }
            all_activity_rows.append(row)
            if molecule_chembl_id:
                seen_molecules.add(molecule_chembl_id)
        time.sleep(args.pause)

    print(f'Fetched {len(all_activity_rows)} activity records; fetching molecule details for {len(seen_molecules)} molecules...')
    for mid in list(seen_molecules):
        try:
            details = fetch_molecule_details(mid, session=session)
            print("Fetched molecule details for", mid)
        except Exception:
            details = {}
        molecule_cache[mid] = details
        time.sleep(args.pause)

    # write CSV
    fieldnames = [
        'assay_chembl_id', 'activity_chembl_id', 'molecule_chembl_id', 'pref_name', 'canonical_smiles',
        'pchembl_value', 'potential_duplicate', 'qudt_units', 'record_id', 'relation', 'src_id',
        'standard_flag', 'standard_relation', 'standard_text_value', 'standard_type', 'standard_units',
        'standard_upper_value', 'standard_value', 'document_chembl_id'
    ]
    with open(args.output, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_activity_rows:
            mid = r.get('molecule_chembl_id')
            details = molecule_cache.get(mid, {}) if mid else {}
            out = {
                **r,
                'pref_name': details.get('pref_name', ''),
                'canonical_smiles': details.get('canonical_smiles', ''),
            }
            writer.writerow(out)

    print('Wrote', args.output)


if __name__ == '__main__':
    main()
