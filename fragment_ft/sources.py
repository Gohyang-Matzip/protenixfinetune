"""Explicit public-data collection and reviewed observation import; stdlib only."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

from .data import (BASE_FIELDS, FIELDS, LABELS, csv_reader, file_hash, row_hash, summary,
                   validate_rows, write_json, write_manifest)

PUBCHEM = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'


def download(url, output, expected_sha256=None, max_bytes=512 * 1024 * 1024):
    """Save exactly the requested resource; keep partial files on failure."""
    if urllib.parse.urlparse(url).scheme not in ('https', 'http') or max_bytes <= 0:
        raise ValueError('Download requires an HTTP(S) URL and positive byte limit')
    output = Path(output)
    partial = output.with_name(output.name + '.partial')
    receipt = output.with_name(output.name + '.receipt.json')
    if any(path.exists() for path in (output, partial, receipt)):
        raise FileExistsError(f'Refusing to overwrite download artifacts: {output}')
    request = urllib.request.Request(url, headers={'User-Agent': 'fragment-ft/0.2 public-research-data'})
    for attempt in range(3):
        try:
            response = urllib.request.urlopen(request, timeout=60)
            break
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
            time.sleep(2 ** attempt)
    digest, size = hashlib.sha256(), 0
    with response, partial.open('xb') as stream:
        expected_length = response.headers.get('Content-Length')
        while block := response.read(1024 * 1024):
            size += len(block)
            if size > max_bytes:
                raise ValueError(f'Download exceeds byte limit; retained {partial}')
            digest.update(block)
            stream.write(block)
        final_url = response.url
        content_type = response.headers.get('Content-Type', '')
    if expected_length is not None and size != int(expected_length):
        raise ValueError(f'Download Content-Length mismatch; retained {partial}')
    checksum = digest.hexdigest()
    if expected_sha256 and checksum.lower() != expected_sha256.lower():
        raise ValueError(f'Download SHA256 mismatch; retained {partial}')
    partial.rename(output)
    result = {'url': url, 'final_url': final_url, 'sha256': checksum, 'bytes': size,
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
        with (output/'compounds.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['source_id', 'ligand_id', 'smiles', 'chem_group'])
            writer.writeheader()
            for start in range(0, len(cids), 100):
                time.sleep(.25)  # Remain below PubChem's five requests/second limit.
                batch = output/f'compounds_{start:07d}.json'
                download(f'{PUBCHEM}/compound/cid/{",".join(cids[start:start+100])}/property/IsomericSMILES/JSON', batch)
                properties = json.loads(batch.read_text())['PropertyTable']['Properties']
                found = {str(item['CID']) for item in properties}
                if found != set(cids[start:start+100]):
                    raise ValueError('PubChem property response omitted requested compounds')
                for item in properties:
                    smiles = item.get('SMILES') or item.get('IsomericSMILES')
                    if not smiles:
                        raise ValueError('PubChem response lacks isomeric SMILES')
                    writer.writerow({'source_id': item['CID'], 'ligand_id': f'CID{item["CID"]}',
                                     'smiles': smiles, 'chem_group': ''})
    return {'aid': int(aid), 'directory': str(output), 'review_required': True,
            'next': 'Review assay/quality and assign chemical groups before import; metadata rows are not observations.'}


def compound_lookup(path):
    if path is None:
        return {}
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv_reader(stream))
    result = {}
    for row in rows:
        key = row.get('source_id') or row.get('ligand_id')
        if not key or key in result:
            raise ValueError('Compound table needs unique source_id (or ligand_id) values')
        result[key] = {k: (row.get(k) or '').strip() for k in ('ligand_id', 'smiles', 'chem_group') if row.get(k)}
    return result


def _import_records(records, config, output, provenance, compounds=None):
    columns, constants = config.get('columns', {}), config.get('constants', {})
    if (set(columns) | set(constants)) - set(FIELDS):
        raise ValueError('Import mapping contains unknown manifest fields')
    if set(columns) & set(constants) or 'label' in columns or 'label' in constants:
        raise ValueError('Use disjoint columns/constants and an explicit outcome mapping for labels')
    outcomes = config['outcomes']
    if not outcomes or not set(outcomes.values()) <= LABELS:
        raise ValueError('outcomes must map source outcomes to 0, 1, uncertain or unknown')
    lookup = compound_lookup(compounds)
    skip = config.get('skip_rows', {})
    if not isinstance(skip, dict) or any(not isinstance(values, list) or
            any(not isinstance(value, str) for value in values) for values in skip.values()):
        raise ValueError('skip_rows must map column names to explicit lists of strings')
    rows, audit = [], []
    skipped = 0
    for number, raw in enumerate(records, start=1):
        if None in raw:
            raise ValueError(f'Row {number}: malformed CSV or unquoted comma')
        audit.append(raw)
        # PubChem CSV may include explicitly named description/type/unit rows.
        if any(str(raw.get(key, '')).strip() in values for key, values in skip.items()):
            skipped += 1
            continue
        try:
            row = {key: str(raw[column] or '').strip() for key, column in columns.items()}
            outcome = str(raw[config['outcome_column']] or '').strip()
        except KeyError as error:
            raise ValueError(f'Row {number}: missing mapped column {error}') from error
        if outcome not in outcomes:
            raise ValueError(f'Row {number}: unmapped outcome {outcome!r}; review it explicitly')
        row.update({key: str(value) for key, value in constants.items()})
        row['label'] = outcomes[outcome]
        if lookup:
            key = row.get('ligand_id')
            if key not in lookup:
                raise ValueError(f'Row {number}: missing compound metadata for {key}')
            for field, value in lookup[key].items():
                if field != 'ligand_id' and row.get(field) and row[field] != value:
                    raise ValueError(f'Row {number}: conflicting compound {field}')
                row[field] = value
        if not row.get('assay_type'):
            raise ValueError('Public imports require explicit assay_type and observation metadata')
        row.setdefault('quality', 'uncertain')
        if 'quality_map' in config:
            if row['quality'] not in config['quality_map']:
                raise ValueError(f'Row {number}: unmapped quality value')
            row['quality'] = config['quality_map'][row['quality']]
        if not row.get('sample_id'):
            row['sample_id'] = 'obs_' + row_hash({'provenance': provenance, 'row': number,
                                                'observation': row})[:24]
        rows.append({key: row.get(key, '') for key in FIELDS if key in BASE_FIELDS or row.get(key)})
    validate_rows(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_manifest(output/'manifest.csv', rows)
    with (output/'raw.jsonl').open('x', encoding='utf-8') as stream:
        for raw in audit:
            stream.write(json.dumps(raw, ensure_ascii=False, allow_nan=False) + '\n')
    write_json(output/'provenance.json', {'inputs': provenance, 'mapping': config,
               'compound_table_sha256': file_hash(compounds) if compounds else None,
               'manifest_sha256': file_hash(output/'manifest.csv'),
               'explicitly_skipped_rows': skipped, 'summary': summary(rows)})
    return rows


def import_csv(path, config, output, compounds=None):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return _import_records(csv_reader(stream), config, output,
                               {str(Path(path).resolve()): file_hash(path)}, compounds)


def import_lit(active, inactive, config, compounds, output):
    """Use supplied outcomes only; SIDs must be joined to curated chemical groups."""
    records = []
    for path, outcome in ((active, 'Active'), (inactive, 'Inactive')):
        with Path(path).open() as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 2:
                    raise ValueError(f'{path}:{number}: expected SMILES and SID')
                records.append({'smiles': fields[0], 'sid': fields[1], 'outcome': outcome})
    mapping = dict(config, columns={'ligand_id': 'sid', 'smiles': 'smiles'},
                   outcome_column='outcome')
    return _import_records(records, mapping, output,
                           {str(Path(p).resolve()): file_hash(p) for p in (active, inactive)}, compounds)
