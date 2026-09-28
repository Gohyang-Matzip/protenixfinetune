"""Explicit public-data collection and reviewed observation import; stdlib only."""
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

from .data import (FIELDS, LABELS, compact, csv_reader, file_hash, observation_key, read_json, row_hash,
                   summary, trainable, validate_rows, write_json, write_manifest)

PUBCHEM = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
MAPPING_KEYS = {'columns', 'constants', 'outcome_column', 'outcomes', 'quality_map', 'skip_rows', 'conflicting_labels'}


class _Redirects(urllib.request.HTTPRedirectHandler):
    """Check each hop before requesting it: stay on HTTP(S) and never leave HTTPS."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlparse(newurl).scheme not in (('https',) if req.type == 'https' else ('https', 'http')):
            fp.close()
            raise ValueError(f'Refusing redirect from {req.full_url} to {newurl}')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url, output, expected_sha256=None, max_bytes=512 * 1024 * 1024):
    """Save exactly the requested resource; keep partial files on failure."""
    if urllib.parse.urlparse(url).scheme not in ('https', 'http') or max_bytes <= 0:
        raise ValueError('Download requires an HTTP(S) URL and positive byte limit')
    output = Path(output)
    partial = output.with_name(output.name + '.partial')
    receipt = output.with_name(output.name + '.receipt.json')
    existing = [str(path) for path in (output, partial, receipt) if path.exists()]
    if existing:
        raise FileExistsError(f'Refusing to overwrite download artifacts {existing}; archive them or choose another output')
    request = urllib.request.Request(url, headers={'User-Agent': 'fragment-ft/0.2 public-research-data'})
    opener = urllib.request.build_opener(_Redirects)
    for attempt in range(3):
        try:
            response = opener.open(request, timeout=60)
            break
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
            time.sleep(2 ** attempt)
    final_url = response.url
    digest, size = hashlib.sha256(), 0
    try:
        with response, partial.open('xb') as stream:
            expected_length = response.headers.get('Content-Length')
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > max_bytes:
                    raise ValueError(f'Download exceeds byte limit; retained {partial}')
                digest.update(block)
                stream.write(block)
            content_type = response.headers.get('Content-Type', '')
    except http.client.HTTPException as error:  # e.g. IncompleteRead on a cut chunked response
        raise ValueError(f'Incomplete download ({error!r}); retained {partial}') from error
    if expected_length is not None and size != int(expected_length):
        raise ValueError(f'Download Content-Length mismatch; retained {partial}')
    checksum = digest.hexdigest()
    if expected_sha256 and checksum.lower() != expected_sha256.lower():
        raise ValueError(f'Download SHA256 mismatch; retained {partial}')
    partial.rename(output)
    # A response without Content-Length and without an expected SHA256 cannot be checked for truncation.
    result = {'url': url, 'final_url': final_url, 'sha256': checksum, 'bytes': size,
              'sha256_verified': bool(expected_sha256),
              'content_length': None if expected_length is None else int(expected_length),
              'content_type': content_type, 'retrieved_utc': datetime.now(timezone.utc).isoformat()}
    write_json(receipt, result)
    return result


def fetch_pubchem(aid, output, with_compounds=False):
    if int(aid) <= 0:
        raise ValueError('PubChem AID must be positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    download(f'{PUBCHEM}/assay/aid/{int(aid)}/description/JSON', output/'description.json')
    download(f'{PUBCHEM}/assay/aid/{int(aid)}/CSV', output/'assay.csv')
    if with_compounds:
        with (output/'assay.csv').open(newline='', encoding='utf-8-sig') as stream:
            reader = csv_reader(stream)
            if 'PUBCHEM_CID' not in (reader.fieldnames or []):
                raise ValueError('PubChem response lacks PUBCHEM_CID; inspect the retained response')
            cids = sorted({row['PUBCHEM_CID'].strip() for row in reader
                           if (row.get('PUBCHEM_CID') or '').strip().isdigit()}, key=int)
        partial = output/'compounds.csv.partial'
        with partial.open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['source_id', 'ligand_id', 'smiles', 'chem_group'])
            writer.writeheader()
            for start in range(0, len(cids), 100):
                time.sleep(.25)  # Remain below PubChem's five requests/second limit.
                batch = output/f'compounds_{start:07d}.json'
                download(f'{PUBCHEM}/compound/cid/{",".join(cids[start:start+100])}/property/IsomericSMILES/JSON', batch)
                properties = read_json(batch)['PropertyTable']['Properties']
                found = {str(item['CID']) for item in properties}
                if found != set(cids[start:start+100]):
                    raise ValueError('PubChem property response omitted requested compounds')
                for item in properties:
                    smiles = item.get('SMILES') or item.get('IsomericSMILES')
                    if not smiles:
                        raise ValueError('PubChem response lacks isomeric SMILES')
                    writer.writerow({'source_id': item['CID'], 'ligand_id': f'CID{item["CID"]}',
                                     'smiles': smiles, 'chem_group': ''})
        partial.rename(output/'compounds.csv')  # Only a complete table gets the final name.
    return {'aid': int(aid), 'directory': str(output), 'review_required': True,
            'next': 'Review assay/quality and assign chemical groups before import; metadata rows are not observations.'}


def compound_lookup(path):
    """Curated compound table keyed by stripped source_id (or ligand_id when there is no source_id column)."""
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv_reader(stream)
        key_column = 'source_id' if 'source_id' in reader.fieldnames else 'ligand_id'
        rows = list(reader)
    if not rows:
        raise ValueError(f'Compound table {path} has no rows')
    result = {}
    for number, row in enumerate(rows, start=1):
        if None in row or None in row.values():
            raise ValueError(f'Compound table row {number}: wrong field count; quote SMILES containing commas')
        key = row.get(key_column, '').strip()
        if not key or key in result:
            raise ValueError(f'Compound table needs unique, nonempty {key_column} values')
        result[key] = {k: row[k].strip() for k in ('ligand_id', 'smiles', 'chem_group') if row.get(k, '').strip()}
    return result


def _strings(mapping):
    return isinstance(mapping, dict) and all(isinstance(v, str) for v in (*mapping, *mapping.values()))


def _import_records(records, fieldnames, config, output, provenance, compounds=None, mapping_sha256=None):
    if not isinstance(config, dict) or set(config) - MAPPING_KEYS:
        raise ValueError(f'Import mapping must be an object using only the keys {sorted(MAPPING_KEYS)}')
    columns, constants = config.get('columns', {}), config.get('constants', {})
    outcome_column, outcomes = config.get('outcome_column'), config.get('outcomes')
    quality_map, skip = config.get('quality_map'), config.get('skip_rows', {})
    policy = config.get('conflicting_labels', 'error')
    if not (_strings(columns) and _strings(constants)):
        raise ValueError('columns and constants must map manifest fields to strings')
    unknown = sorted((set(columns) | set(constants)) - set(FIELDS))
    if unknown:
        raise ValueError(f'Import mapping contains unknown manifest fields: {unknown}')
    if set(columns) & set(constants) or 'label' in columns or 'label' in constants:
        raise ValueError('Use disjoint columns/constants and an explicit outcome mapping for labels')
    if not isinstance(outcome_column, str):
        raise ValueError('outcome_column must name the source outcome column')
    if not _strings(outcomes) or not outcomes or not set(outcomes.values()) <= LABELS:
        raise ValueError('outcomes must map source outcomes to 0, 1, uncertain or unknown')
    if quality_map is not None and not _strings(quality_map):
        raise ValueError('quality_map must map source QC values to pass, fail or uncertain')
    if not isinstance(skip, dict) or any(not isinstance(values, list) or
            any(not isinstance(value, str) for value in values) for values in skip.values()):
        raise ValueError('skip_rows must map column names to explicit lists of strings')
    if policy not in ('error', 'uncertain'):
        raise ValueError("conflicting_labels must be 'error' or 'uncertain'")
    missing = sorted({outcome_column, *columns.values(), *skip} - set(fieldnames))
    if missing:
        raise ValueError(f'Mapped columns are not in the input header: {missing}')
    lookup = None if compounds is None else compound_lookup(compounds)
    rows, audit, skipped = [], [], 0
    for number, raw in enumerate(records, start=1):
        if None in raw or None in raw.values():
            raise ValueError(f'Row {number}: wrong field count; malformed CSV or unquoted comma')
        audit.append({'row': number, 'sample_id': None, 'raw': raw})
        # PubChem CSV may include explicitly named description/type/unit rows.
        if any(raw[key].strip() in values for key, values in skip.items()):
            skipped += 1
            continue
        outcome = raw[outcome_column].strip()
        if outcome not in outcomes:
            raise ValueError(f'Row {number}: unmapped outcome {outcome!r}; review it explicitly')
        row = {key: raw[column].strip() for key, column in columns.items()}
        row.update({key: value.strip() for key, value in constants.items()})
        row['label'] = outcomes[outcome]
        if lookup is not None:
            key = row.get('ligand_id', '')
            if key not in lookup:
                raise ValueError(f'Row {number}: missing compound metadata for {key!r}')
            for field, value in lookup[key].items():
                if field != 'ligand_id' and row.get(field) and row[field] != value:
                    raise ValueError(f'Row {number}: conflicting compound {field}')
                row[field] = value
        if not row.get('assay_type'):
            raise ValueError('Public imports require explicit assay_type and observation metadata')
        if 'quality' not in row:
            row['quality'] = 'uncertain'  # Missing QC is never promoted to pass; quality_map applies to mapped QC only.
        elif quality_map is not None:
            if row['quality'] not in quality_map:
                raise ValueError(f'Row {number}: unmapped quality value {row["quality"]!r}')
            row['quality'] = quality_map[row['quality']]
        row = compact(row)
        if not row['sample_id']:  # File content, not location, identifies the input.
            row['sample_id'] = 'obs_' + row_hash({'inputs': sorted(provenance.values()), 'row': number,
                                                'observation': row})[:24]
        audit[-1]['sample_id'] = row['sample_id']
        rows.append(row)
    observed = defaultdict(list)
    for row in rows:
        if trainable(row):
            observed[observation_key(row)].append(row['label'])
    conflicting = {key for key, labels in observed.items() if len(set(labels)) > 1}
    if conflicting and policy == 'error':
        raise ValueError(f'{len(conflicting)} observation(s) have both 0 and 1 labels, e.g. {min(conflicting)}; '
                         'review them, or set "conflicting_labels": "uncertain" to hold them out')
    for row in rows:  # Never resolved toward a negative: every conflicting observation leaves training.
        if trainable(row) and observation_key(row) in conflicting:
            row['label'] = 'uncertain'
    validate_rows(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_manifest(output/'manifest.csv', rows)
    with (output/'raw.jsonl').open('x', encoding='utf-8') as stream:
        for record in audit:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    write_json(output/'provenance.json', {'inputs': provenance, 'mapping': config, 'mapping_sha256': mapping_sha256,
               'compound_table_sha256': None if compounds is None else file_hash(compounds),
               'manifest_sha256': file_hash(output/'manifest.csv'), 'explicitly_skipped_rows': skipped,
               'conflicting_labels': policy, 'replicate_keys': sum(len(labels) > 1 for labels in observed.values()),
               'conflicting_keys': len(conflicting), 'summary': summary(rows)})
    return rows


def import_csv(path, config, output, compounds=None, mapping_sha256=None):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv_reader(stream)
        return _import_records(reader, reader.fieldnames, config, output, {str(path): file_hash(path)},
                               compounds, mapping_sha256)


def import_lit(active, inactive, config, compounds, output, mapping_sha256=None):
    """Use supplied outcomes only; SIDs must be joined to curated chemical groups."""
    records = []
    for path, outcome in ((active, 'Active'), (inactive, 'Inactive')):
        with Path(path).open(encoding='utf-8') as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 2:
                    raise ValueError(f'{path}:{number}: expected SMILES and SID')
                records.append({'smiles': fields[0], 'sid': fields[1], 'outcome': outcome})
    # Also catches one file passed twice (hard link, copy): a listed active must never become a negative,
    # whatever the outcome mapping or conflicting_labels policy.
    active_sids, inactive_sids = ({r['sid'] for r in records if r['outcome'] == o} for o in ('Active', 'Inactive'))
    both = active_sids & inactive_sids
    if both:
        raise ValueError(f'{len(both)} SID(s) listed as both active and inactive, e.g. {sorted(both)[:5]}; '
                         'review the lists')
    mapping = dict(config, columns={'ligand_id': 'sid', 'smiles': 'smiles'},
                   outcome_column='outcome') if isinstance(config, dict) else config
    return _import_records(records, ('smiles', 'sid', 'outcome'), mapping, output,
                           {str(p): file_hash(p) for p in (active, inactive)}, compounds, mapping_sha256)
