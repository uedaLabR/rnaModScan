"""Worker-local indexed BAM fetches retain exact sampled data and input order."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow.parquet as pq

from tests import test_grouped_bam_features
from utils.features.site_bag_pipeline import extract_site_bags_and_index
from utils.features.parallel_groups import merge_group_reports, prepare_jobs


def stored_sites(directory):
    report = json.loads((directory / 'run.json').read_text())
    rows = []
    for item in report['parquet_files']['sites']:
        rows.extend(pq.ParquetFile(directory / 'sites' / item['path']).read().to_pylist())
    return {row['row_id']: row for row in rows}


class ParallelGroupsTest(unittest.TestCase):
    def test_parallel_matches_serial_sampling_and_has_ordered_index(self):
        self.compare_parallel(2)

    def test_three_workers_preserve_sampling_and_order(self):
        self.compare_parallel(3)

    def test_merged_fetch_batches_preserve_sampling_and_order(self):
        self.compare_parallel(3, batch_size=2)

    def test_four_workers_with_span_limit_preserve_all_sampled_data(self):
        self.compare_parallel(4, batch_size=16, max_fetch_span=5000)

    def compare_parallel(self, workers, batch_size=1, max_fetch_span=10000000):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sites, bam, fasta = test_grouped_bam_features.GroupedBamFeaturesTest().write_fixture(root)
            with sites.open() as handle:
                reader = csv.DictReader(handle, delimiter='\t')
                fields = reader.fieldnames
                rows = [dict(row, modification='adenine') for row in reader if row['modification'] == 'm6A']
            with sites.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fields, delimiter='\t')
                writer.writeheader()
                writer.writerows(rows)
            sample = dict(sample_id='sample', cell='HEK293T', reference_cell='HEK293T', condition='WT', bam=bam)
            options = dict(mode='inference', max_reads=1, flank=2, candidate_group_gap=10000,
                           group_batch_size=batch_size, max_fetch_span=max_fetch_span)
            serial = extract_site_bags_and_index(sites, [sample], fasta, root / 'serial', None, **options)
            parallel = extract_site_bags_and_index(sites, [sample], fasta, root / 'parallel', None,
                group_workers=workers, **options)
            self.assertEqual(serial['counts'], parallel['counts'])
            expected = stored_sites(root / 'serial/samples/sample')
            actual = stored_sites(root / 'parallel/samples/sample')
            self.assertEqual(expected, actual)
            with (root / 'parallel/index/adenine/sites.tsv').open() as handle:
                rows = list(csv.DictReader(handle, delimiter='\t'))
            keys = [(row['chrom'], int(row['start']), row['row_id']) for row in rows]
            self.assertEqual(keys, sorted(keys))
            self.assertEqual(len(keys), len(set(keys)))
            report = json.loads((root / 'parallel/samples/sample/run.json').read_text())
            group_count = 6 if max_fetch_span == 5000 else 5
            self.assertEqual(report['counts']['candidate_groups_completed'], group_count)
            self.assertFalse((root / 'parallel/samples/sample/INCOMPLETE.txt').exists())
            self.assertEqual(report['group_workers'], workers)
            expected_fetches = 6 if max_fetch_span == 5000 else (5 + batch_size - 1) // batch_size
            self.assertEqual(report['counts']['bam_fetches'], expected_fetches)

    def test_task_batches_merge_strands_and_stop_at_chromosome_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sites = root / 'sites.tsv'
            sites.write_text('sample_id\tchrom\tstart\tstrand\n'
                             'sample\tchr1\t100\t+\n'
                             'sample\tchr1\t100\t-\n'
                             'sample\tchr2\t100\t-\n'
                             'sample\tchr2\t10100\t+\n')
            options = dict(candidate_group_gap=10000, flank=2)
            jobs, total_sites, total_groups = prepare_jobs(
                sites, dict(sample_id='sample'), 'unused.fa', root, options, 16)
            self.assertEqual((len(jobs), total_sites, total_groups), (2, 4, 3))
            for job, chrom in zip(jobs, ('chr1', 'chr2')):
                with Path(job['site_list']).open() as handle:
                    rows = list(csv.DictReader(handle, delimiter='\t'))
                self.assertEqual([row['chrom'] for row in rows], [chrom, chrom])

    def test_merge_uses_input_order_even_when_files_were_completed_in_reverse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = [dict(out=str(root / 'sites' / f'batch-{index:06d}')) for index in range(3)]
            for index in (2, 1, 0):
                directory = Path(jobs[index]['out'])
                directory.mkdir(parents=True)
                report = dict(status='complete', counts=dict(sites=1, bam_records_total=10),
                    observed_mm_groups={'m6A': 1}, modification_channels=['m6A'], policies={},
                    parquet_files=dict(sites=[dict(path=f'chrom=chr1/strand=plus/{index}.parquet')]))
                (directory / 'run.json').write_text(json.dumps(report))
                (directory / 'site_summary.tsv').write_text(f'row_id\tdepth\nsite{index}\t10\n')
            report = merge_group_reports(jobs, root, {}, {}, 2, 1, 0)
            self.assertEqual((root / 'site_summary.tsv').read_text().splitlines(),
                             ['row_id\tdepth', 'site0\t10', 'site1\t10', 'site2\t10'])
            self.assertEqual([item['path'].split('/')[0] for item in report['parquet_files']['sites']],
                             ['batch-000000', 'batch-000001', 'batch-000002'])
            self.assertEqual(report['counts']['bam_records_total'], 10)
            self.assertEqual(report['counts']['sites'], 3)

    def test_group_parallelism_rejects_nested_sample_workers(self):
        with self.assertRaisesRegex(ValueError, 'sample_workers=1'):
            extract_site_bags_and_index('unused', [dict(sample_id='sample')], 'unused', 'unused',
                                       None, mode='inference', sample_workers=2, group_workers=2)


if __name__ == '__main__':
    unittest.main()
