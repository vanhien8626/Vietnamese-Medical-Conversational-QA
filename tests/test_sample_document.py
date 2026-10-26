import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock, patch

from read_document import main


class SampleDocumentTests(unittest.TestCase):
    def test_sample_loads_without_paste_from_another_directory(self):
        reader = Mock()
        reader.answer.return_value = {'answer': 'Dengue', 'score': 0.8, 'sources': []}
        previous = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                with patch('read_document.ReadingModel', return_value=reader), \
                        patch('builtins.input', side_effect=AssertionError('Should not prompt for document')), \
                        redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(main(['--sample', '--question', 'Virus nào?', '--json']), 0)
        finally:
            os.chdir(previous)
        document = reader.answer.call_args.args[0]
        self.assertGreater(len(document), 10000)
        self.assertIn('PHÒNG NGỪA TRONG GIA ĐÌNH VÀ CỘNG ĐỒNG', document)
        self.assertNotIn('/xong', document)

    def test_sample_and_document_are_mutually_exclusive(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(['--sample', '--document', 'any.txt'])
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
