"""Joint model wiring on synthetic Parquet/BAM, not biological accuracy."""
import csv
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ('tensorflow', 'pyarrow', 'pysam'))


@unittest.skipUnless(AVAILABLE, 'TensorFlow/Arrow/pysam required')
class AdenineModelTest(unittest.TestCase):
    def save_bundle(self, directory, position_source=None, max_reads=300):
        import numpy as np
        from test_inference import input_schema
        from prediction.predict import _load_model
        from utils.nn.keras_site import ModelConfig, build_site_model
        from utils.adenine import CLASSES
        from utils.references.catalog import sha256
        directory.mkdir(parents=True)
        config = ModelConfig(target='adenine', max_reads=max_reads)
        model = build_site_model(config)
        layer = model.get_layer('class_scores')
        kernel, bias = layer.get_weights()
        layer.set_weights([np.zeros_like(kernel), np.log(np.array([.2, .7, .1], dtype=bias.dtype))])
        model.save(directory / 'model.keras')
        schema = input_schema()
        schema.update(class_names=list(CLASSES), max_reads=max_reads, sampling_seed=42)
        run = dict(status='complete', target='adenine', role='final_all_labeled_sites',
                   model=asdict(config), model_file='model.keras', model_sha256=sha256(directory / 'model.keras'),
                   threshold=.5,
                   input_schema=schema, position_annotation=position_source)
        (directory / 'run.json').write_text(json.dumps(run))
        self.assertEqual(_load_model(directory)[0].output_shape, (None, 3))

    def test_three_class_parquet_training_cv_and_final_model(self):
        from tests.site_bag_fixture import prepare_training_bags
        from training.cross_validation import build_cross_validation
        from training.fit import train_fold
        from training.final_model import train_final_model
        from utils.features.site_dataset import SiteBagBatches as SiteBatches
        from prediction.predict import _load_model
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = prepare_training_bags(root)
            data = SiteBatches(index, None, batch_sites=4)
            inputs, targets = next(data.batches())
            self.assertEqual(set(inputs), {'sequence_tokens', 'read_features', 'read_mask', 'sequence_repeat_mask'})
            self.assertEqual(targets.shape, (4,))
            self.assertEqual(str(targets.dtype), 'int32')
            build_cross_validation({'adenine': index}, root / 'cv')
            report = train_fold(index, root / 'cv/adenine/fold_0', root / 'models/fold_0', 'adenine',
                epochs=1, batch_sites=8)
            self.assertEqual(set(report['class_weights']), {0, 1, 2})
            metrics = json.loads((root / 'models/fold_0/test/metrics.json').read_text())
            self.assertEqual(len(metrics['confusion_matrix']), 3)
            for fold in range(1, 5):
                shutil.copytree(root / 'models/fold_0', root / f'models/fold_{fold}')
            final = train_final_model(index, root / 'models', root / 'final', 'adenine', epochs=1,
                batch_sites=8)
            self.assertEqual(final['training_rows'], 48)
            self.assertEqual(_load_model(root / 'final')[0].output_shape, (None, 3))

    def test_public_joint_prediction_independent_flags_and_bam_output(self):
        from array import array
        import pysam
        import yaml
        from rnamodscan.cli import main
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequence = list('A' * 100)
            sequence[8:13] = 'AGACA'
            sequence = ''.join(sequence)
            fasta = root / 'genome.fa'
            fasta.write_text('>chr1\n' + sequence + '\n')
            pysam.faidx(str(fasta))
            bam = root / 'reads.bam'
            with pysam.AlignmentFile(str(bam), 'wb', header=dict(HD=dict(SO='coordinate'), SQ=[dict(SN='chr1', LN=100)])) as handle:
                for pos in (10, 30, 50, 70):
                    for i in range(10):
                        read = pysam.AlignedSegment(handle.header)
                        read.query_name = f'{pos}:{i}'
                        read.query_sequence = sequence[pos-9:pos+10]
                        read.query_qualities = array('B', [30]*19)
                        read.flag, read.reference_id, read.reference_start = 0, 0, pos-9
                        read.cigarstring, read.mapping_quality = '19M', 60
                        delta = read.query_sequence[:9].count('A')
                        read.set_tag('MM', f'A+a?,{delta};A+17596?,{delta};')
                        read.set_tag('ML', array('B', [220,200]))
                        handle.write(read)
            pysam.index(str(bam))
            (root / 'calls.bed').write_text(''.join(
                f'chr1 {pos} {pos+1} {code} 10 + {pos} {pos+1} 0,0,0 10 30 3 4 3 0 0 0 0\n'
                for pos in (10,30,50,70) for code in ('a','17596')))
            (root / 'pileup.log').write_text('command line: synthetic\n')
            sources = []
            for target, positions in [('m6A',(30,50)), ('inosine',(50,))]:
                path = root / (target + '.bed')
                path.write_text(''.join(f'chr1\t{pos}\t{pos+1}\tx\t0\t+\n' for pos in positions))
                sources.append(dict(id=target, kind='database', database='RMBase', version='3.0',
                    modification=target, cell='*', assembly='GRCh38', coordinates='BED0', path=path.name))
            (root / 'catalog.json').write_text(json.dumps(dict(assembly='GRCh38', sources=sources)))
            self.save_bundle(root / 'models/adenine', max_reads=300)
            config = dict(schema_version=1, mode='predict', output_dir='output', targets=['m6A','inosine'],
                model_root='models', references=dict(fasta='genome.fa', catalog='catalog.json'),
                sample=dict(sample_id='sample', cell='HEK293T', reference_cell='HEK293T', condition='WT',
                            bam='reads.bam', bedmethyl='calls.bed', pileup_log='pileup.log'),
                nn=dict(task='adenine_three_class'),
                filtering=dict(m6a_drach_keep=True))
            path = root / 'config.yaml'
            path.write_text(yaml.safe_dump(config))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['predict','--config',str(path),'--recalibrate']),0)
            with (root / 'output/predictions/adenine/all_classifications.tsv').open() as handle:
                rows = list(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual(len(rows),4)
            self.assertEqual([row['final_class'] for row in rows], ['m6A;inosine','m6A;inosine','m6A;inosine','inosine'])
            self.assertTrue(all(row['predicted_class']=='inosine' for row in rows))
            with (root / 'output/predictions/adenine/class_counts.tsv').open() as handle:
                self.assertEqual(sum(int(row['sites']) for row in csv.DictReader(handle, delimiter='\t')),4)
            bed = root / 'output/predictions/bedmethyl/filtered.bed'
            returned = [line.split() for line in bed.read_text().splitlines() if not line.startswith('#')]
            self.assertEqual(len(returned), 7)
            self.assertTrue(all(len(row) == 23 and row[-1] == '1' for row in returned))
            by_site = {(int(row[1]), row[3]): row for row in returned}
            self.assertEqual(by_site[10, 'a'][-5:], ['0', '0', '0', '1', '1'])
            self.assertEqual(by_site[30, 'a'][-5:], ['1', '0', '0', '0', '1'])
            self.assertEqual(by_site[50, '17596'][-5:], ['1', '0', '1', '0', '1'])
            self.assertNotIn((70, 'a'), by_site)
            raw = json.loads((root / 'output/features/samples/sample/run.json').read_text())
            self.assertEqual(raw['counts']['sites'],4)  # Eight input rows produce four read bags.
            with pysam.AlignmentFile(str(root / 'output/bam/filtered.bam'),'rb') as handle:
                for read in handle.fetch(until_eof=True):
                    pos = int(read.query_name.split(':')[0])
                    self.assertEqual(list(read.get_tag('ML')), [220,200] if pos in (10,30,50) else [0,200])


if __name__ == '__main__':
    unittest.main()
