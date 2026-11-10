import csv
from pathlib import Path
import tempfile
import unittest

from prediction.candidates import prepare_prediction_sites
from prediction.confidence import AUDIT_FIELDS, BASE_FIELDS, combine_confidence, confidence_fields
from utils.io.bed import parse_bedmethyl_row


class PredictionConfidenceTest(unittest.TestCase):
    def pileup(self, position, *, modified=3, valid=10, failed=10, different=5, nocall=5):
        return (f'chr1 {position} {position+1} a {valid} + {position} {position+1} 0,0,0 '
                f'{valid} 30 {modified} {valid-modified} 0 0 {failed} {different} {nocall}\n')

    def test_confidence_filtered_fraction_differs_from_unchanged_all_read_af(self):
        row = parse_bedmethyl_row(self.pileup(10), 1)
        audit = confidence_fields(row)
        self.assertEqual(row['modified_reads'], 3)  # Already a modkit passing-call count.
        self.assertEqual(row['total_reads'], 30)
        self.assertEqual(row['af'], .1)
        self.assertEqual(audit, dict(valid_coverage=10, failed_calls=10,
                                    different_base_reads=5, no_calls=5, af_pass=.3))
        # Extra failed-confidence calls change legacy AF, not the passing-call AF.
        more_failed = parse_bedmethyl_row(self.pileup(20, failed=30), 2)
        self.assertEqual(more_failed['af'], .06)
        self.assertEqual(confidence_fields(more_failed)['af_pass'], .3)

    def test_zero_valid_coverage_is_undefined_and_missing_target_is_not_zero(self):
        row = parse_bedmethyl_row(self.pileup(10, modified=0, valid=0), 1)
        audit = confidence_fields(row)
        self.assertEqual(audit['valid_coverage'], 0)
        self.assertEqual(audit['af_pass'], '')
        result = dict(af=.1)
        self.assertIs(combine_confidence(result, {'m6A': audit}), result)
        self.assertEqual(result['af'], .1)
        self.assertEqual(result['valid_coverage_m6A'], 0)
        self.assertEqual(result['af_pass_m6A'], '')
        self.assertEqual(result['valid_coverage_inosine'], '')
        self.assertEqual(result['af_pass_inosine'], '')
        self.assertEqual(AUDIT_FIELDS[:len(BASE_FIELDS)], BASE_FIELDS)

    def test_candidate_selection_still_uses_existing_af_and_carries_audit_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'calls.bed'
            source.write_text(self.pileup(10) + self.pileup(20, failed=30))
            log = root / 'pileup.log'
            log.write_text('setting threshold 0.7 (qual: 179) for base A, percentile 10\n')
            settings = {'m6A': dict(min_depth=10, min_modified=3, min_af=.1)}
            report = prepare_prediction_sites(source, log, 'sample', root / 'out', settings)
            with (root / 'out/m6A/sites.tsv').open() as handle:
                rows = list(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual(len(rows), 1)  # Position20 has af_pass=.3 but legacy AF=.06.
            self.assertEqual(rows[0]['start'], '10')
            self.assertEqual(rows[0]['af'], '0.1')
            self.assertEqual(rows[0]['af_pass'], '0.3')
            self.assertEqual(rows[0]['valid_coverage'], '10')
            self.assertEqual(rows[0]['failed_calls'], '10')
            self.assertEqual(rows[0]['different_base_reads'], '5')
            self.assertEqual(rows[0]['no_calls'], '5')
            self.assertIn('audit only', report['af_definitions']['af_pass'])


if __name__ == '__main__':
    unittest.main()
