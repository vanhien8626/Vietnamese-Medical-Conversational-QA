import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluate import evaluate_dataset
from rag_engine import DocumentIndex

ROOT = Path(__file__).resolve().parents[1]


class AdvancedWorkflowTests(unittest.TestCase):
    def test_multiple_gold_answers_and_abstention_counts(self):
        report = evaluate_dataset([{
            'context': 'Người bệnh nên nghỉ ngơi.',
            'turns': [
                {'question': 'Người bệnh nên làm gì?',
                 'answer': ['Nghỉ ngơi.', 'Người bệnh nên nghỉ ngơi.']},
                {'question': 'Thiên hà Andromeda?', 'answer': 'Không có thông tin.'},
            ],
        }])
        self.assertEqual(report['exact_match'], 0.5)
        self.assertEqual(report['unanswered'], 1)
        self.assertEqual(report['conversation_reports'][0]['questions'], 2)
        self.assertFalse(report['conversation_reports'][0]['consistent'])

    def test_invalid_gold_answer_lists(self):
        for gold in ([], [''], ['hợp lệ', None], {'text': 'x'}):
            with self.subTest(gold=gold), self.assertRaises(ValueError):
                evaluate_dataset([{'context': 'Tài liệu.', 'turns': [{'question': 'Gì?', 'answer': gold}]}])

    def test_sparse_search_matches_full_scan(self):
        document = ('Viêm phổi do vi khuẩn gây ra. Triệu chứng gồm ho và sốt. '
                    'Người bệnh cần nghỉ ngơi. Trường hợp nặng gây biến chứng. '
                    'Tiêm vắc xin giúp phòng ngừa. Sốt cao cần được theo dõi.')
        sparse = DocumentIndex(document)
        full = DocumentIndex(document)
        # Ép duyệt mọi câu làm đối chứng, bao gồm từ khóa không có trong tài liệu.
        class FullPostings(dict):
            def get(self, key, default=None):
                return set(range(len(full.sentences)))
        full._postings = FullPostings()
        for question in ('Triệu chứng là gì?', 'Nguy hiểm không?', 'Nó do nguyên nhân gì?',
                         'Sốt cao?', 'Thiên hà Andromeda?', 'Người bệnh cần làm gì?'):
            for threshold in (0, 0.15, 0.7):
                with self.subTest(question=question, threshold=threshold):
                    history = [{'question': 'Viêm phổi có triệu chứng gì?'}]
                    self.assertEqual(sparse.search(question, history, threshold=threshold),
                                     full.search(question, history, threshold=threshold))

    def test_cli_export_candidates_clear_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory) / 'document.txt'
            document.write_text('Người bệnh nên nghỉ ngơi. Nghỉ ngơi giúp phục hồi.', encoding='utf-8')
            output = Path(directory) / 'session.json'
            command = [sys.executable, '-B', str(ROOT / 'chat.py'), '--document', str(document),
                       '--top-k', '2', '--export', str(output)]
            result = subprocess.run(command, input='Nghỉ ngơi?\n/lichsu\n/xoa\nNgười bệnh nên làm gì?\n/thoat\n',
                                    capture_output=True, text=True, encoding='utf-8', timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(len(report['turns']), 1)
            self.assertEqual(report['turns'][0]['question'], 'Người bệnh nên làm gì?')
            self.assertEqual(report['top_k'], 2)
            self.assertIn('candidates', report['turns'][0])
            before = output.read_bytes()
            repeated = subprocess.run(command, input='/thoat\n', capture_output=True, text=True, timeout=15)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(before, output.read_bytes())


if __name__ == '__main__':
    unittest.main()
