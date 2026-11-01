import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock, patch

from rag_engine import DocumentIndex
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

    def test_symptom_question_retrieves_the_actual_symptom_list(self):
        from pathlib import Path
        document = (Path(__file__).resolve().parents[1] / 'data/dengue_document.txt').read_text()
        passages = DocumentIndex(document).search(
            'Sốt xuất huyết có triệu chứng gì?', top_k=3)

        self.assertTrue(any(
            passage['text'].startswith('Các triệu chứng thường gặp của sốt xuất huyết')
            for passage in passages
        ))


if __name__ == '__main__':
    unittest.main()
