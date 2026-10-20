import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from reader_engine import NO_ANSWER, ReaderError, ReadingModel
from read_document import main

DOCUMENT = 'Người bệnh nên nghỉ ngơi và uống đủ nước.'


def controlled_reader(result):
    reader = ReadingModel.__new__(ReadingModel)
    reader.model = 'test-model'
    reader._qa = Mock(return_value=result)
    return reader


class ReaderTests(unittest.TestCase):
    def test_answer_comes_from_original_document(self):
        start = DOCUMENT.index('nghỉ ngơi')
        reader = controlled_reader({'answer': 'nghỉ ngơi', 'start': start,
                                    'end': start + len('nghỉ ngơi'), 'score': 0.8})
        result = reader.answer(DOCUMENT, 'Người bệnh nên làm gì?')
        self.assertEqual(DOCUMENT[result['start']:result['end']], result['answer'])
        self.assertIn(result['answer'], result['sources'][0])
        self.assertTrue(reader._qa.call_args.kwargs['handle_impossible_answer'])
        self.assertEqual(reader._qa.call_args.kwargs['doc_stride'], 128)
        self.assertEqual(reader._qa.call_args.kwargs['context'], DOCUMENT)

    def test_low_score_and_empty_model_answer_abstain(self):
        for output in ({'answer': 'nghỉ ngơi', 'score': 0.01}, {'answer': '', 'score': 0.9}):
            result = controlled_reader(output).answer(DOCUMENT, 'Làm gì?')
            self.assertEqual(result['answer'], NO_ANSWER)
            self.assertEqual(result['sources'], [])
            self.assertIsNone(result['start'])

    def test_invalid_model_offsets_and_scores_are_rejected(self):
        for output in ({'answer': 'bịa', 'score': 0.8, 'start': 0, 'end': 3},
                       {'answer': 'x', 'score': float('nan')},
                       {'answer': 'x', 'score': 1.5}, None):
            with self.subTest(output=output), self.assertRaises(ReaderError):
                controlled_reader(output).answer(DOCUMENT, 'Làm gì?')

    def test_invalid_input_does_not_call_model(self):
        reader = controlled_reader(None)
        for threshold in (-1, 2, float('nan')):
            with self.assertRaises(ValueError):
                reader.answer(DOCUMENT, 'Làm gì?', threshold=threshold)
        with self.assertRaises(ValueError):
            reader.answer('', 'Làm gì?')
        reader._qa.assert_not_called()

    def test_inference_failure_is_reported_without_internal_details(self):
        reader = controlled_reader(None)
        reader._qa.side_effect = RuntimeError('internal detail')
        with self.assertRaises(ReaderError) as error:
            reader.answer(DOCUMENT, 'Làm gì?')
        self.assertNotIn('internal detail', str(error.exception))

    def test_cli_pasted_document_and_json_stdout(self):
        reader = controlled_reader({'answer': 'nghỉ ngơi', 'score': 0.7,
                                    'start': DOCUMENT.index('nghỉ ngơi'),
                                    'end': DOCUMENT.index('nghỉ ngơi') + len('nghỉ ngơi')})
        with patch('read_document.ReadingModel', return_value=reader), \
                patch('builtins.input', side_effect=[DOCUMENT, '/xong']), \
                redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--question', 'Nên làm gì?', '--json']), 0)
        self.assertEqual(json.loads(out.getvalue())['answer'], 'nghỉ ngơi')

    def test_cli_document_and_missing_model_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tài liệu.txt'
            path.write_text(DOCUMENT, encoding='utf-8-sig')
            with patch('read_document.ReadingModel', side_effect=ReaderError('Chưa tải mô hình.')), \
                    redirect_stderr(io.StringIO()) as out:
                self.assertEqual(main(['--document', str(path), '--question', 'Làm gì?']), 1)
            self.assertIn('Chưa tải mô hình.', out.getvalue())


if __name__ == '__main__':
    unittest.main()
