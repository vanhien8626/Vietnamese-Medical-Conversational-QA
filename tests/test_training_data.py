import json
import tempfile
import unittest
from pathlib import Path

from train_reader import load_rows, question_with_history


class TrainingDataTests(unittest.TestCase):
    def test_load_rows_validates_answer_offsets(self):
        context = 'Sốt xuất huyết do virus Dengue gây ra.'
        row = {
            'question': 'Virus nào?',
            'context': context,
            'answers': {
                'text': ['Dengue'],
                'answer_start': [context.index('Dengue')],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.jsonl'
            path.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
            self.assertEqual(load_rows(path), [row])
            row['answers']['answer_start'] = [0]
            path.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_rows(path)

    def test_history_is_bounded_and_current_question_is_last(self):
        row = {
            'question': 'Nó lây thế nào?',
            'history': [
                {'question': 'Câu 1?', 'answer': 'Đáp án 1.'},
                {'question': 'Câu 2?', 'answer': 'Đáp án 2.'},
            ],
        }
        result = question_with_history(row, max_history=1)
        self.assertNotIn('Câu 1?', result)
        self.assertIn('Câu 2?', result)
        self.assertTrue(result.endswith('Câu hỏi hiện tại: Nó lây thế nào?'))

