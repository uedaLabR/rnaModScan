"""Parallel BAM isolation, exact feature reuse and completed bag release."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from array import array
from collections import Counter
from io import StringIO
from unittest.mock import Mock


@unittest.skipUnless(importlib.util.find_spec('pysam') and importlib.util.find_spec('pyarrow'), 'BAM dependencies missing')
class ParallelExtractionTest(unittest.TestCase):
    def test_reused_query_features_equal_uncached_windows_on_both_strands(self):
        import pysam
        from utils.features.bam import encode_read, MODIFICATIONS
        from utils.io.cigar import CigarIndex
        class Modifications(dict):
            lookups = 0
            def get(self, key, default=None):
                self.lookups += 1
                return super().get(key, default)
        for reverse in (False, True):
            read = pysam.AlignedSegment()
            read.query_name = 'overlapping'
            read.reference_start = 10
            read.flag = 16 if reverse else 0
            read.query_sequence = 'ACGT' * 40
            read.query_qualities = array('B', [30] * 160)
            read.cigarstring = '160M'
            index = CigarIndex(read)
            mods = Modifications({(q, mod): (q + i) % 256 for q in range(160) for i, mod in enumerate(MODIFICATIONS)})
            cache, cached, uncached = {}, [], []
            for position in (60, 61, 62):
                site = dict(start=position, strand='-' if reverse else '+')
                mapping, insertions = index.window(position - 9, position + 10)
                cached.append(encode_read(read, site, 9, mapping, insertions, mods, {}, 'A' * 19,
                                          index.junctions, base_cache=cache))
            cached_lookups = mods.lookups
            mods.lookups = 0
            for position in (60, 61, 62):
                site = dict(start=position, strand='-' if reverse else '+')
                mapping, insertions = index.window(position - 9, position + 10)
                uncached.append(encode_read(read, site, 9, mapping, insertions, mods, {}, 'A' * 19, index.junctions))
            self.assertEqual(cached, uncached)
            self.assertEqual(cached_lookups, 21 * 8)
            self.assertEqual(mods.lookups, 3 * 19 * 8)

    def test_completed_site_bag_is_cleared(self):
        from utils.features.bam import write_site_features
        site = dict(row_id='site', sample_id='s', chrom='chr1', start=10, end=11, strand='+', modification='m6A')
        read = dict(mm_skip_modes={}, modification_observed=[[False] * 8] * 19)
        retained = Mock()
        retained.features_for_site.return_value = read
        bag = dict(reads=[retained], reference='A' * 19, deletions=0, skips=0)
        reads, sites, counts = Mock(), Mock(), Counter()
        fasta = Mock()
        fasta.get_reference_length.return_value = 1000
        fasta.fetch.return_value = 'A' * 19
        write_site_features(site, bag, fasta, 9, reads, sites, csv.writer(StringIO(), delimiter='\t'), counts)
        self.assertEqual(bag['reads'], [])
        self.assertEqual(counts['retained_read_site_pairs'], 1)

    def test_two_bams_use_distinct_worker_processes_and_match_serial_outputs(self):
        import pysam
        import pyarrow.parquet as pq
        from utils.features.site_bag_pipeline import extract_site_bags_and_index
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fasta = root / 'reference.fa'
            fasta.write_text('>chr1\n' + 'A' * 200 + '\n>chr2\n' + 'A' * 200 + '\n')
            pysam.faidx(str(fasta))
            header = dict(HD=dict(SO='coordinate'), SQ=[dict(SN=chrom, LN=200) for chrom in ('chr1', 'chr2')])
            samples = []
            for cell in ('HEK293T', 'HeLa'):
                bam = root / (cell + '.bam')
                with pysam.AlignmentFile(str(bam), 'wb', header=header) as out:
                    for contig in (0, 1):
                        read = pysam.AlignedSegment(out.header)
                        read.query_name, read.query_sequence = cell + str(contig), 'A' * 50
                        read.reference_id, read.reference_start, read.flag = contig, 10, 0
                        read.cigarstring = '50M'
                        read.query_qualities = array('B', [30] * 50)
                        read.set_tag('MM', 'A+a?,10,9;')
                        read.set_tag('ML', array('B', [200, 100]))
                        out.write(read)
                pysam.index(str(bam))
                samples.append(dict(sample_id=cell + '_WT', cell=cell, reference_cell=cell,
                                    condition='WT', bam=bam))
            source = root / 'candidates.tsv'
            fields = ['row_id', 'sample_id', 'chrom', 'start', 'end', 'strand', 'modification']
            with source.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fields, delimiter='\t')
                writer.writeheader()
                for chrom in ('chr1', 'chr2'):
                    for position in (20, 21, 30):
                        for sample in samples:
                            writer.writerow(dict(row_id=f'{sample["sample_id"]}|{chrom}:{position}',
                                sample_id=sample['sample_id'], chrom=chrom, start=position, end=position + 1,
                                strand='+', modification='adenine'))
            extract_site_bags_and_index(source, samples, fasta, root / 'serial', None, mode='inference')
            extract_site_bags_and_index(source, samples, fasta, root / 'parallel', None,
                                        mode='inference', sample_workers=2, bam_threads=2)
            pids = set()
            for sample in samples:
                left = root / 'serial/samples' / sample['sample_id']
                right = root / 'parallel/samples' / sample['sample_id']
                records = []
                for directory in (left, right):
                    run = json.loads((directory / 'run.json').read_text())
                    rows = []
                    for item in run['parquet_files']['sites']:
                        rows.extend(pq.ParquetFile(directory / 'sites' / item['path']).read().to_pylist())
                    records.append(sorted(rows, key=lambda row: row['row_id']))
                self.assertEqual(records[0], records[1])
                progress = json.loads((right / 'progress.json').read_text())
                self.assertEqual(progress['status'], 'complete')
                self.assertEqual(progress['counts']['sites'], 6)
                pids.add(progress['pid'])
            self.assertEqual(len(pids), 2)
            self.assertNotIn(os.getpid(), pids)
