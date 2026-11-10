import csv
import json
from pathlib import Path
import tempfile
import unittest

from training.adenine_sites import select_adenine_sites, labeled_sites
from prediction.adenine_filtering import retained_adenine_classes


class AdenineSitesTest(unittest.TestCase):
    def source(self, root):
        source = root / 'sites'
        source.mkdir()
        (source / 'run.json').write_text(json.dumps(dict(status='complete', window_flank=9)))
        fields = ['row_id', 'sample_id', 'cell', 'condition', 'chrom', 'start', 'end', 'strand',
                  'modification', 'af', 'label', 'label_status', 'label_reason']
        # Labels: one m6A, one I, one conflict, one WT both-negative, one WT partly unknown.
        specs = dict(m6A=[(10,'WT','1'), (20,'WT','0'), (30,'WT','1'), (40,'WT','0'), (50,'WT','0'),
                         (60,'IVT','0'), (70,'IVT','0'), (80,'IVT','0'), (90,'IVT','0')],
                     inosine=[(10,'WT',''), (20,'WT','1'), (30,'WT','1'), (40,'WT','0'), (60,'IVT','0')])
        for target, records in specs.items():
            folder = source / target
            folder.mkdir()
            with (folder / 'all_sites.tsv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fields, delimiter='\t')
                writer.writeheader()
                for start, condition, label in records:
                    sample = 'cell_' + condition
                    writer.writerow(dict(row_id=f'{sample}|chr1:{start}:+:{target}', sample_id=sample,
                        cell='cell', condition=condition, chrom='chr1', start=start, end=start+1,
                        strand='+', modification=target, af=.3, label=label, label_status='final',
                        label_reason='ivt_candidate_and_paired' if start==60 else 'fixture'))
        return source

    def test_merge_labels_unknowns_ambiguity_pairing_and_reproducible_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.source(root)
            paths = [source / target / 'all_sites.tsv' for target in ('m6A','inosine')]
            rows = {int(row['start']): row for row in labeled_sites(paths)}
            self.assertEqual(rows[10]['label'], '0')
            self.assertEqual(rows[10]['af_inosine'], '0.3')  # Unlabeled candidates still supply metadata.
            self.assertEqual(rows[20]['label'], '1')
            self.assertEqual(rows[30]['label_reason'], 'both_positive_excluded')
            self.assertEqual(rows[40]['label'], '2')
            self.assertEqual(rows[50]['label_reason'], 'other_target_unknown')
            self.assertEqual(rows[60]['label_reason'], 'paired_ivt')
            run = select_adenine_sites(source, root / 'selected')
            self.assertEqual(run['selected_sites'], 5)  # Two positives, paired IVT, two extra FP.
            self.assertEqual(run['excluded'], dict(both_positive_excluded=1, other_target_unknown=1))
            with (root / 'selected/adenine/training_sites.tsv').open() as handle:
                selected = list(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual(sum(row['selection_reason']=='random_negative' for row in selected), 2)
            self.assertEqual(len({row['row_id'] for row in selected}), 5)
            self.assertIn(60, [int(row['start']) for row in selected])
            self.assertEqual(select_adenine_sites(source, root / 'selected'), run)

    def test_independent_or_preserves_only_original_candidates(self):
        scores = dict(score_m6A=.2, score_inosine=.3, score_false_positive=.5, predicted_class='false_positive')
        candidates = ['m6A', 'inosine']
        self.assertEqual(retained_adenine_classes(scores, ['m6A'], False, candidates), ['m6A'])
        self.assertEqual(retained_adenine_classes(scores, ['inosine'], True, candidates), candidates)
        self.assertEqual(retained_adenine_classes(scores, candidates, False, candidates), candidates)
        self.assertEqual(retained_adenine_classes(scores, [], True, candidates), ['m6A'])
        self.assertEqual(retained_adenine_classes(scores, [], False, candidates), [])
        self.assertEqual(retained_adenine_classes(None, candidates, False, candidates), candidates)
        self.assertEqual(retained_adenine_classes(scores, ['inosine'], True, ['m6A']), ['m6A'])



if __name__ == '__main__':
    unittest.main()
