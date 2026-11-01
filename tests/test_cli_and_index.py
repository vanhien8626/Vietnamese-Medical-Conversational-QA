import json
import subprocess
import sys
import tempfile
import unicodedata
import unittest
from pathlib import Path

from evaluate import evaluate_dataset
from rag_engine import DocumentIndex, answer_question

ROOT = Path(__file__).resolve().parents[1]


class IndexTests(unittest.TestCase):
    def test_decomposed_vietnamese_matches_composed_text(self):
        document = 'Triệu chứng thường gặp gồm sốt cao và đau đầu.'
        question = 'Triệu chứng là gì?'
        expected = answer_question(document, question)
        self.assertEqual(answer_question(unicodedata.normalize('NFD', document),
                                         unicodedata.normalize('NFD', question)), expected)

    def test_reference_word_is_not_matched_inside_another_word(self):
        index = DocumentIndex('Sốt nóng gây khó chịu. Người bệnh cần nghỉ ngơi.')
        history = [{'question': 'Người bệnh cần nghỉ ngơi?'}]
        self.assertEqual(index.search('Sốt nóng?', history), index.search('Sốt nóng?'))

    def test_repeated_queries_do_not_change_index_or_history(self):
        index = DocumentIndex('Người bệnh cần nghỉ ngơi. Sốt là triệu chứng thường gặp.')
        history = [{'question': 'Sốt là gì?', 'answer': 'Sốt là triệu chứng thường gặp.'}]
        previous = list(history)
        first = index.answer('Người bệnh cần làm gì?', history)
        index.answer('Sốt là gì?')
        self.assertEqual(index.answer('Người bệnh cần làm gì?', history), first)
        self.assertEqual(history, previous)
        self.assertEqual(first['answer'], 'Người bệnh cần nghỉ ngơi.')

    def test_ranking_and_invalid_parameters(self):
        index = DocumentIndex('Người bệnh cần nghỉ ngơi. Nghỉ ngơi giúp phục hồi.')
        matches = index.search('Nghỉ ngơi?', top_k=2)
        self.assertEqual(len(matches), 2)
        self.assertGreaterEqual(matches[0]['score'], matches[1]['score'])
        for threshold in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                index.search('Nghỉ ngơi?', threshold=threshold)
        for top_k in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                index.search('Nghỉ ngơi?', top_k=top_k)
        for options in ({'k1': 0}, {'k1': float('nan')}, {'b': -0.1}, {'b': 1.1}):
            with self.assertRaises(ValueError):
                DocumentIndex('Tài liệu.', **options)

    def test_bm25_length_normalization_prefers_focused_sentence(self):
        index = DocumentIndex(
            'Sốt cao. '
            'Sốt là một biểu hiện được nhắc đến trong câu rất dài có nhiều từ không liên quan.'
        )
        matches = index.search('Sốt?', top_k=2, threshold=0)

        self.assertEqual(matches[0]['text'], 'Sốt cao.')
        self.assertIn('bm25_score', matches[0])

    def test_subject_match_beats_unrelated_intent_and_reference_url(self):
        index = DocumentIndex(
            'Điều trị viêm phổi phụ thuộc vào nguyên nhân. '
            'Sốt xuất huyết do virus Dengue gây ra. '
            'CDC, Phòng ngừa: https://example.com/reference'
        )
        result = index.search('Nguyên nhân gây sốt xuất huyết là gì?', top_k=3)

        self.assertEqual(result[0]['text'], 'Sốt xuất huyết do virus Dengue gây ra.')
        self.assertFalse(any('https://' in item['text'] for item in result))

    def test_route_and_avoid_medicine_queries_rank_actionable_sentence(self):
        document = (
            'Muỗi Aedes có thể truyền virus. '
            'Bệnh lây qua vết đốt của muỗi Aedes. '
            'Giảm tiểu cầu có thể gặp ở người mắc bệnh. '
            'Không tự dùng aspirin hoặc ibuprofen khi nghi mắc bệnh.'
        )
        index = DocumentIndex(document)

        route = index.search('Muỗi Aedes truyền virus qua cách nào?', top_k=1)
        medicine = index.search('Người nghi mắc bệnh cần tránh thuốc nào?', top_k=1)

        self.assertEqual(route[0]['text'], 'Bệnh lây qua vết đốt của muỗi Aedes.')
        self.assertIn('aspirin hoặc ibuprofen', medicine[0]['text'])

    def test_evaluation_rejects_empty_and_malformed_data(self):
        for data in ([], {}, [None], [{'context': 'a', 'turns': []}],
                     [{'context': 'a', 'turns': [{'question': 'b'}]}]):
            with self.subTest(data=data), self.assertRaises(ValueError):
                evaluate_dataset(data)


class CLITests(unittest.TestCase):
    def run_cli(self, directory, script, *args, stdin=None):
        return subprocess.run([sys.executable, '-B', str(ROOT / script), *map(str, args)],
                              cwd=directory, input=stdin, text=True, encoding='utf-8',
                              capture_output=True, timeout=15)

    def test_document_json_and_interactive_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory) / 'văn bản.txt'
            document.write_text('Người bệnh nên nghỉ ngơi.', encoding='utf-8-sig')
            result = self.run_cli(directory, 'chat.py', '--document', document,
                                  '--question', 'Người bệnh nên làm gì?', '--json')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['answer'], 'Người bệnh nên nghỉ ngơi.')
            interactive = self.run_cli(directory, 'chat.py', '--document', document,
                                       stdin='Người bệnh nên làm gì?\n/xoa\n/thoat\n')
            self.assertEqual(interactive.returncode, 0, interactive.stderr)
            self.assertIn('Đã xóa lịch sử.', interactive.stdout)

    def test_missing_file_and_invalid_arguments_fail_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            for args in (('--document', 'missing.txt'), ('--question', ''),
                         ('--threshold', 'nan'), ('--json',)):
                result = self.run_cli(directory, 'chat.py', *args)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('Traceback', result.stderr)

    def test_evaluation_report_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            result = self.run_cli(directory, 'evaluate.py', '--output', output)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(report['questions'], 32)
            self.assertEqual(len(report['predictions']), 32)
            original = output.read_bytes()
            repeated = self.run_cli(directory, 'evaluate.py', '--output', output)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(output.read_bytes(), original)
            bad = Path(directory) / 'bad.json'
            bad.write_text('[]', encoding='utf-8')
            invalid = self.run_cli(directory, 'evaluate.py', '--data', bad)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertNotIn('Traceback', invalid.stderr)


if __name__ == '__main__':
    unittest.main()
