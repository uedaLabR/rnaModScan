"""Selective reads retain numeric/blank values while splitting scientific books."""
from pathlib import Path
import tempfile
import unittest

from utils.io.excel import verify_workbook, write_workbook


class ExcelSubsetTest(unittest.TestCase):
    def test_selected_sheet_values_match_complete_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'data.xlsx'
            write_workbook(path, [dict(name='Summary',headers=['n','fraction','note'],rows=[[0,.125,None],[42,.5,'=literal']]),
                                  dict(name='Raw',headers=['position'],rows=[[10],[20]])])
            all_values = verify_workbook(path)
            selected = verify_workbook(path, sheet_names=['Summary'])
            self.assertEqual(selected, {'Summary':all_values['Summary']})
            self.assertEqual(selected['Summary'][1], [0,.125,None])
            self.assertEqual(selected['Summary'][2][-1], '=literal')
            with self.assertRaisesRegex(ValueError, 'missing'):
                verify_workbook(path, sheet_names=['Unknown'])


if __name__=='__main__':
    unittest.main()
