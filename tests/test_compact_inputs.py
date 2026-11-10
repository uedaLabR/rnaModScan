"""Integer codes restore exactly the float32 values of the unchanged NN inputs."""
import unittest

import numpy as np
import pyarrow as pa

from utils.features.compact_inputs import integer_read_type, pack_read_inputs, restore_read_inputs
from utils.nn.inputs import encode_read_row


class CompactInputsTest(unittest.TestCase):
    def rows(self):
        rng = np.random.RandomState(42)
        bases = [None, 'A', 'C', 'G', 'T', 'N']
        result = []
        for index in range(100):
            result.append(dict(base=[bases[rng.randint(6)] for _ in range(19)],
                ml_byte=[[None if rng.randint(4) == 0 else int(rng.randint(256)) for _ in range(8)] for _ in range(19)],
                base_quality=[None if rng.randint(4) == 0 else int(rng.randint(256)) for _ in range(19)],
                insertion_length_after=[int(rng.randint(1000000)) for _ in range(19)],
                mismatch=[None if rng.randint(3) == 0 else int(rng.randint(2)) for _ in range(19)],
                splice_distance_before=None if index % 3 else index * 400,
                splice_distance_after=index * 31))
        return result

    def test_exact_round_trip_and_integer_widths(self):
        rows = self.rows()
        rows[0]['ml_byte'][0] = [0, 255, None, 128, 1, 254, 99, 100]
        rows[0]['base_quality'][0:3] = [0, 255, None]
        rows[0]['insertion_length_after'][0] = np.iinfo(np.uint32).max
        packed = pack_read_inputs(rows)
        value = pa.array([packed], type=pa.list_(integer_read_type(19)))[0]
        actual = restore_read_inputs(value, 19)
        expected = np.asarray([encode_read_row(row) for row in rows], dtype=np.float32)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(value.values.field('ml_byte').type.value_type, pa.uint8())
        self.assertEqual(packed[0]['ml_byte'][:3], [0, 255, 0])
        self.assertEqual(packed[0]['quality_code'][:3], [1, 256, 0])
        self.assertIsNone(rows[0]['ml_byte'][0][2])  # Raw audit is untouched.

    def test_all_base_categories_absence_and_clipped_splice(self):
        row = dict(base=[None, 'A', 'C', 'G', 'T', 'N'], mismatch=[None, 0, 1, None, 1, 0],
                   ml_byte=[[None] * 8 for _ in range(6)], base_quality=[None] * 6,
                   insertion_length_after=[0] * 6, splice_distance_before=None, splice_distance_after=100000)
        packed = pack_read_inputs([row])
        self.assertEqual(packed[0]['base_category'], [0, 1, 3, 4, 5, 6])
        self.assertEqual((packed[0]['splice_before'], packed[0]['splice_after']), (1000, 1000))
        value = pa.array([packed], type=pa.list_(integer_read_type(6)))[0]
        np.testing.assert_array_equal(restore_read_inputs(value, 6), np.asarray([encode_read_row(row)], np.float32))
        empty = pa.array([[]], type=pa.list_(integer_read_type(6)))[0]
        self.assertEqual(restore_read_inputs(empty, 6).shape, (0, 6, 18))

    def test_invalid_codes_and_float32_legacy_storage(self):
        rows = self.rows()[:1]
        expected = np.asarray([encode_read_row(rows[0])], np.float32)
        legacy_type = pa.list_(pa.list_(pa.list_(pa.float32(), 18), 19))
        legacy = pa.array([expected.tolist()], type=legacy_type)[0]
        np.testing.assert_array_equal(restore_read_inputs(legacy, 19), expected)
        packed = pack_read_inputs(rows)
        packed[0]['quality_code'][0] = 257
        with self.assertRaisesRegex(ValueError, 'Invalid integer'):
            restore_read_inputs(pa.array([packed], type=pa.list_(integer_read_type(19)))[0], 19)
        rows[0]['ml_byte'][0][0] = 256
        with self.assertRaises(ValueError):
            pack_read_inputs(rows)


if __name__ == '__main__':
    unittest.main()
