"""Public-data download and import guards; stdlib only, no network."""
import email.message
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
import urllib.request
import urllib.response

from fragment_ft import data, sources

CONSTANTS = {'target_id': 'T1', 'target_group': 'F1', 'assay_id': 'screen', 'assay_type': 'direct_binding',
             'endpoint': 'binding', 'campaign_id': 'study', 'source': 'example',
             'source_url': 'https://example.org/study'}
SCREEN = 'id,smiles,group,outcome,qc\n1,CC,G1,Active,OK\n2,CO,G2,Inactive,Failed\n'


def mapping(**changes):
    config = {'columns': {'ligand_id': 'id', 'smiles': 'smiles', 'chem_group': 'group', 'quality': 'qc'},
              'constants': dict(CONSTANTS), 'outcome_column': 'outcome',
              'outcomes': {'Active': '1', 'Inactive': '0', 'Inconclusive': 'uncertain'},
              'quality_map': {'OK': 'pass', 'Failed': 'fail'}}
    config.update(changes)
    return config


def serve(routes):
    """Local HTTP stub: path -> body bytes, or a callable(handler)."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = routes.get(self.path)
            if callable(body):
                return body(self)
            self.send_response(200 if body is not None else 404)
            if body is not None and not self.path.startswith('/nolength'):
                self.send_header('Content-Length', str(len(body)))
            self.end_headers(); self.wfile.write(body or b'')
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    return server, worker


class SourceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def http(self, routes):
        server, worker = serve(routes)
        self.addCleanup(lambda: (server.shutdown(), worker.join(), server.server_close()))
        return f'http://127.0.0.1:{server.server_port}'

    def write(self, name, text):
        (self.root/name).write_text(text)
        return self.root/name

    def run_import(self, name, text, config, **kwargs):
        return sources.import_csv(self.write(f'{name}.csv', text), config, self.root/name, **kwargs)

    def provenance(self, name):
        return json.loads((self.root/name/'provenance.json').read_text())

    def test_missing_qc_is_uncertain_even_with_quality_map(self):
        config = mapping(); del config['columns']['quality']
        plain = {k: v for k, v in config.items() if k != 'quality_map'}
        for name, cfg in (('plain', plain), ('mapped', config)):
            rows = self.run_import(name, SCREEN, cfg)
            self.assertEqual([r['quality'] for r in rows], ['uncertain', 'uncertain'])
            self.assertFalse(any(data.trainable(r) for r in rows))

    def test_quality_map_translates_and_rejects_unreviewed_values(self):
        rows = self.run_import('qc', SCREEN, mapping())
        self.assertEqual([(r['label'], r['quality']) for r in rows], [('1', 'pass'), ('0', 'fail')])
        with self.assertRaisesRegex(ValueError, "unmapped quality value 'Pending'"):
            self.run_import('pending', SCREEN + '3,CN,G3,Inactive,Pending\n', mapping())
        self.assertFalse((self.root/'pending').exists())

    def test_malformed_mappings_fail_before_writing(self):
        columns = mapping()['columns']
        cases = [({'columns': dict(columns, label='outcome')}, 'explicit outcome mapping'),
                 ({'outcomes': {'Active': 'active'}}, 'outcomes'),
                 ({'outcomes': ['Active']}, 'outcomes'),
                 ({'columns': dict(columns, smiles=['smiles'])}, 'strings'),
                 ({'constants': dict(CONSTANTS, source_url=None)}, 'strings'),
                 ({'outcome_column': None}, 'outcome_column'),
                 ({'quality_map': ['OK']}, 'quality_map'),
                 ({'columns': dict(columns, chem_group='grp')}, r"header: \['grp'\]"),
                 ({'skip_rows': {'TAG': ['RESULT_TYPE']}}, r"header: \['TAG'\]"),
                 ({'skip_rows': {'qc': 'OK'}}, 'skip_rows'),
                 ({'conflicting_labels': 'drop'}, 'conflicting_labels'),
                 ({'quality_mapp': {}}, 'only the keys'),
                 ({'columns': dict(columns, Smiles='smiles')}, 'unknown manifest fields')]
        for number, (change, message) in enumerate(cases):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                self.run_import(f'bad{number}', SCREEN, mapping(**change))
            self.assertFalse((self.root/f'bad{number}').exists())

    def test_short_and_long_rows_are_rejected(self):
        for name, row in (('short', '3,CN,G3,Inactive\n'), ('long', '3,CN,G3,Inactive,OK,extra\n')):
            with self.subTest(name), self.assertRaisesRegex(ValueError, 'Row 3: wrong field count'):
                self.run_import(name, SCREEN + row, mapping())

    def test_compound_table_guards(self):
        screen = 'id,outcome,qc\n101,Active,OK\n'
        config = mapping(columns={'ligand_id': 'id', 'quality': 'qc'})
        header = 'source_id,ligand_id,smiles,chem_group\n'
        rows = self.run_import('join', screen, config, compounds=self.write('ok.csv', header + '101 ,L1,CC,G1\n'))
        self.assertEqual([(r['ligand_id'], r['smiles'], r['chem_group']) for r in rows], [('L1', 'CC', 'G1')])
        cases = [(header + ',101,CN,G9\n', 'nonempty source_id'),
                 (header + '101,L1,CC,G1\n101,L2,CO,G2\n', 'unique'),
                 (header + '101,Cmpd 5, batch 2,CCO,G1\n', 'row 1: wrong field count'),
                 (header + '101,L1,CC\n', 'row 1: wrong field count'),
                 (header, 'no rows'),
                 (header + '102,L2,CO,G2\n', "missing compound metadata for '101'")]
        for number, (table, message) in enumerate(cases):
            with self.subTest(table=table), self.assertRaisesRegex(ValueError, message):
                self.run_import(f'bad{number}', screen, config, compounds=self.write(f'c{number}.csv', table))
        with self.assertRaisesRegex(ValueError, 'conflicting compound smiles'):
            self.run_import('conflict', SCREEN.replace('2,CO', '101,CO'), mapping(),
                            compounds=self.write('c.csv', header + '1,L1,CC,G1\n101,L2,CN,G2\n'))

    def test_conflicting_labels_are_refused_or_held_out_never_negative(self):
        text = ('id,smiles,group,outcome,qc\n1,CC,G1,Active,OK\n1,CC,G1,Inactive,OK\n2,CO,G2,Inactive,OK\n'
                '2,CO,G2,Inactive,OK\n3,CN,G3,Active,OK\n1,CC,G1,Inactive,Failed\n')
        with self.assertRaisesRegex(ValueError, 'conflicting_labels'):
            self.run_import('strict', text, mapping())
        rows = self.run_import('held', text, mapping(conflicting_labels='uncertain'))
        self.assertEqual([(r['label'], r['quality']) for r in rows],  # the QC-fail row is not relabeled
                         [('uncertain', 'pass'), ('uncertain', 'pass'), ('0', 'pass'), ('0', 'pass'), ('1', 'pass'),
                          ('0', 'fail')])
        report = self.provenance('held')
        self.assertEqual((report['conflicting_labels'], report['replicate_keys'], report['conflicting_keys']),
                         ('uncertain', 2, 1))

    def test_auto_ids_follow_content_and_provenance_keeps_given_paths(self):
        config = mapping(constants=dict(CONSTANTS, campaign_id=' study '))
        imported = []
        for name in ('a', 'b'):
            (self.root/name).mkdir()
            source = os.path.relpath(self.write(f'{name}/screen.csv', SCREEN))  # kept as given, not resolved
            imported.append(sources.import_csv(source, config, self.root/f'{name}_out', mapping_sha256='f' * 64))
            report = self.provenance(f'{name}_out')
            self.assertEqual(list(report['inputs']), [str(source)])
            self.assertEqual(report['mapping_sha256'], 'f' * 64)
        self.assertEqual([r['sample_id'] for r in imported[0]], [r['sample_id'] for r in imported[1]])
        self.assertEqual(imported[0][0]['campaign_id'], 'study')
        self.assertEqual(data.read_manifest(self.root/'a_out/manifest.csv'), imported[0])

    def test_lit_refuses_sids_in_both_lists_under_any_mapping(self):
        compounds = self.write('compounds.csv', 'source_id,ligand_id,smiles,chem_group\n101,L1,CC,G1\n102,L2,CO,G2\n')
        config = {'constants': dict(CONSTANTS, quality='pass'), 'outcomes': {'Active': 'uncertain', 'Inactive': '0'}}
        active = self.write('active.smi', 'CC 101\n')
        os.link(active, self.root/'linked.smi')
        inactives = [f'{self.root}/./active.smi', self.root/'linked.smi', self.write('copy.smi', 'CC 101\n'),
                     self.write('inactive.smi', 'CO 102\nCC 101\n')]
        for number, inactive in enumerate(inactives):
            for cfg in (config, dict(config, conflicting_labels='uncertain'),
                        dict(config, outcomes={'Active': '1', 'Inactive': '0'})):
                with self.subTest(inactive=inactive, cfg=cfg), \
                        self.assertRaisesRegex(ValueError, r"both active and inactive.*\['101'\]"):
                    sources.import_lit(active, inactive, cfg, compounds, self.root/f'out{number}')
                self.assertFalse((self.root/f'out{number}').exists())
        rows = sources.import_lit(active, self.write('clean.smi', 'CO 102\n'), config, compounds, self.root/'clean')
        self.assertEqual([(r['ligand_id'], r['label'], data.trainable(r)) for r in rows],
                         [('L1', 'uncertain', False), ('L2', '0', True)])

    def test_download_receipt_errors_and_existing_artifacts(self):
        def chunked(handler):
            handler.send_response(200); handler.send_header('Transfer-Encoding', 'chunked')
            handler.end_headers(); handler.wfile.write(b'a\r\nhello')
        base = self.http({'/data': b'public data', '/nolength': b'half', '/chunked': chunked})
        digest = hashlib.sha256(b'public data').hexdigest()
        receipt = sources.download(f'{base}/data', self.root/'data', digest)
        self.assertEqual((receipt['sha256_verified'], receipt['content_length']), (True, 11))
        receipt = sources.download(f'{base}/nolength', self.root/'nolength')
        self.assertEqual((receipt['sha256_verified'], receipt['content_length']), (False, None))
        with self.assertRaisesRegex(ValueError, 'Incomplete download.*retained'):
            sources.download(f'{base}/chunked', self.root/'cut')
        self.assertTrue((self.root/'cut.partial').exists())
        self.assertFalse((self.root/'cut').exists())
        with self.assertRaisesRegex(FileExistsError, 'cut.partial'):
            sources.download(f'{base}/data', self.root/'cut')

    def test_download_checks_each_redirect_hop_before_requesting_it(self):
        hits = []
        def redirect(location):
            def handler(h):
                hits.append(h.path); h.send_response(302); h.send_header('Location', location)
                h.send_header('Content-Length', '0'); h.end_headers()
            return handler
        base = self.http({'/ftp': redirect('ftp://127.0.0.1:9/file'), '/hop': redirect('https://elsewhere.example/x')})
        def https_open(handler, request):  # an HTTPS origin that redirects to plaintext
            headers = email.message.Message(); headers['Location'] = f'{base}/hop'
            response = urllib.response.addinfourl(io.BytesIO(), headers, request.full_url, 302)
            response.msg = 'Found'
            return response
        with self.assertRaisesRegex(ValueError, 'Refusing redirect .*/ftp to ftp:'):
            sources.download(f'{base}/ftp', self.root/'ftp')
        with mock.patch.object(urllib.request.HTTPSHandler, 'https_open', https_open):
            with self.assertRaisesRegex(ValueError, 'Refusing redirect from https://origin.example/f to http:'):
                sources.download('https://origin.example/f', self.root/'file')
        self.assertEqual(hits, ['/ftp'])  # the plaintext hop was never requested
        self.assertEqual(list(self.root.iterdir()), [])

    def test_fetch_pubchem_builds_compounds_and_imports_with_skip_rows(self):
        assay = ('PUBCHEM_RESULT_TAG,PUBCHEM_SID,PUBCHEM_CID,PUBCHEM_ACTIVITY_OUTCOME\n'
                 'RESULT_TYPE,,,\n1,501,2,Active\n2,502,10,Inactive\n').encode()
        properties = {'PropertyTable': {'Properties': [{'CID': 2, 'SMILES': 'CC'}, {'CID': 10, 'IsomericSMILES': 'CO'}]}}
        base = self.http({'/assay/aid/7/description/JSON': b'{}', '/assay/aid/7/CSV': assay,
                          '/compound/cid/2,10/property/IsomericSMILES/JSON': json.dumps(properties).encode(),
                          '/assay/aid/8/description/JSON': b'{}', '/assay/aid/8/CSV': assay.replace(b',10,', b',11,'),
                          '/compound/cid/2,11/property/IsomericSMILES/JSON': json.dumps(
                              {'PropertyTable': {'Properties': properties['PropertyTable']['Properties'][:1]}}).encode()})
        with mock.patch.object(sources, 'PUBCHEM', base):
            sources.fetch_pubchem(7, self.root/'aid7', with_compounds=True)
            with self.assertRaisesRegex(ValueError, 'omitted'):
                sources.fetch_pubchem(8, self.root/'aid8', with_compounds=True)
        self.assertEqual((self.root/'aid7/compounds.csv').read_text().splitlines(),
                         ['source_id,ligand_id,smiles,chem_group', '2,CID2,CC,', '10,CID10,CO,'])
        self.assertFalse((self.root/'aid7/compounds.csv.partial').exists())
        self.assertFalse((self.root/'aid8/compounds.csv').exists())
        self.assertTrue((self.root/'aid8/compounds.csv.partial').exists())
        reviewed = self.write('reviewed.csv', 'source_id,ligand_id,smiles,chem_group\n2,CID2,CC,G1\n10,CID10,CO,G2\n')
        config = {'columns': {'ligand_id': 'PUBCHEM_CID'}, 'constants': dict(CONSTANTS, quality='pass'),
                  'outcome_column': 'PUBCHEM_ACTIVITY_OUTCOME', 'outcomes': {'Active': '1', 'Inactive': '0'},
                  'skip_rows': {'PUBCHEM_RESULT_TAG': ['RESULT_TYPE']}}
        rows = sources.import_csv(self.root/'aid7/assay.csv', config, self.root/'imported', reviewed)
        self.assertEqual([(r['ligand_id'], r['smiles'], r['label']) for r in rows], [('CID2', 'CC', '1'), ('CID10', 'CO', '0')])
        self.assertEqual(self.provenance('imported')['explicitly_skipped_rows'], 1)
        raw = [json.loads(line) for line in (self.root/'imported/raw.jsonl').read_text().splitlines()]
        self.assertEqual([(r['row'], r['sample_id'], r['raw']['PUBCHEM_SID']) for r in raw],
                         [(1, None, ''), (2, rows[0]['sample_id'], '501'), (3, rows[1]['sample_id'], '502')])


if __name__ == '__main__':
    unittest.main()
