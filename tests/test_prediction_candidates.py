import csv
import tempfile
import unittest
from pathlib import Path

from prediction.candidates import prepare_prediction_sites


class PredictionCandidatesTest(unittest.TestCase):
    def test_modkit_candidates_have_no_labels_and_use_shared_af(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'calls.bed'
            source.write_text('chr1 10 11 a 10 + 10 11 0,0,0 10 30 3 7 0 100 10 5 5\n'
                              'chr1 20 21 a 10 - 20 21 0,0,0 10 20 2 8 0 0 0 0 0\n'
                              'chr1 30 31 19228 10 + 30 31 0,0,0 10 30 3 7 0 0 0 0 0\n')
            log = root / 'pileup.log'
            log.write_text('setting threshold 0.7 (qual: 179) for base A, percentile 0.1\nDone, processed 3 rows\n')
            thresholds = {m: dict(min_depth=10, min_modified=3, min_af=.1) for m in ('m6A', 'm5C')}
            report = prepare_prediction_sites(source, log, 'new_sample', root / 'out', thresholds)
            with (root / 'out/m6A/sites.tsv').open() as handle:
                rows = list(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['af'], '0.1')
            self.assertEqual(rows[0]['label'], '')
            self.assertEqual(rows[0]['row_id'], 'new_sample|chr1:10:+:m6A')
            self.assertEqual(report['upstream']['filter_mode'], 'sampled_per_base')
            with (root / 'out/candidate_counts.tsv').open() as handle:
                counts = {r['modification']: r['candidate_sites'] for r in csv.DictReader(handle, delimiter='\t')}
            self.assertEqual(counts['m5C'], '0')
