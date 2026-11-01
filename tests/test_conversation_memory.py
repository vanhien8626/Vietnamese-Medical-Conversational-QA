import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock, patch

from conversation_memory import ConversationMemory, infer_topic
from read_document import main


class ConversationMemoryTests(unittest.TestCase):
    def test_topic_inference_and_reference_rewrite(self):
        memory = ConversationMemory()
        first = 'Sốt xuất huyết do virus nào gây ra?'
        memory.record(first, first, 'Virus Dengue.')

        resolved, used = memory.resolve('Nó lây truyền như thế nào?')

        self.assertTrue(used)
        self.assertEqual(resolved, 'Sốt xuất huyết lây truyền như thế nào?')
        self.assertEqual(infer_topic('Viêm phổi có triệu chứng gì?'), 'Viêm phổi')

    def test_clear_and_bounded_history(self):
        memory = ConversationMemory(max_turns=2)
        for number in range(3):
            question = f'Bệnh {number} có nguy hiểm không?'
            memory.record(question, question, 'Có.')
        self.assertEqual(len(memory.turns), 2)
        memory.clear()
        self.assertEqual(memory.turns, [])
        self.assertIsNone(memory.topic)
        self.assertEqual(memory.resolve('Nó là gì?'), ('Nó là gì?', False))

    def test_contextualize_uses_bounded_history(self):
        memory = ConversationMemory(max_turns=3)
        memory.record('Câu một?', 'Câu một?', 'Đáp án một.')
        memory.record('Câu hai?', 'Câu hai?', 'Đáp án hai.')

        contextual = memory.contextualize('Nó lây thế nào?', max_history=1)

        self.assertNotIn('Câu một?', contextual)
        self.assertIn('Câu hỏi trước: Câu hai?', contextual)
        self.assertTrue(contextual.endswith('Câu hỏi hiện tại: Nó lây thế nào?'))

    def test_interactive_reader_uses_resolved_followup_and_commands(self):
        backend = Mock()
        backend.answer.side_effect = [
            {'answer': 'Dengue', 'score': 0.8, 'sources': []},
            {'answer': 'Qua muỗi Aedes', 'score': 0.7, 'sources': []},
        ]
        inputs = iter([
            'Sốt xuất huyết do virus nào gây ra?',
            'Nó lây truyền như thế nào?',
            '/lichsu',
            '/xoa',
            '/thoat',
        ])
        with patch('read_document.RetrieverReader', return_value=backend), \
                patch('builtins.input', side_effect=lambda *args: next(inputs)), \
                redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--sample', '--mode', 'retriever-reader']), 0)

        self.assertEqual(backend.answer.call_args_list[1].args[0],
                         'Sốt xuất huyết lây truyền như thế nào?')
        self.assertIn('Hiểu theo ngữ cảnh: Sốt xuất huyết lây truyền như thế nào?', output.getvalue())
        self.assertIn('Đã xóa lịch sử và chủ đề hội thoại.', output.getvalue())

    def test_output_labels_original_source_and_answer_offsets_clearly(self):
        backend = Mock()
        backend.answer.return_value = {
            'answer': 'Dengue',
            'score': 0.8,
            'sources': ['Sốt xuất huyết do virus Dengue gây ra.'],
            'start': 28,
            'end': 34,
        }
        with patch('read_document.ReadingModel', return_value=backend), \
                redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()):
            self.assertEqual(main([
                '--sample', '--question', 'Virus nào gây bệnh?'
            ]), 0)

        text = output.getvalue()
        self.assertIn('Đoạn văn gốc dùng để trả lời:', text)
        self.assertIn('“Sốt xuất huyết do virus Dengue gây ra.”', text)
        self.assertIn('Vị trí đáp án trong tài liệu: ký tự 28–34', text)

    def test_concat_mode_sends_history_only_to_reader(self):
        backend = Mock()
        backend.answer.side_effect = [
            {'answer': 'Dengue', 'score': 0.8, 'sources': []},
            {'answer': 'Muỗi Aedes', 'score': 0.8, 'sources': []},
        ]
        inputs = iter([
            'Sốt xuất huyết do virus nào?',
            'Nó lây thế nào?',
            '/thoat',
        ])
        with patch('read_document.RetrieverReader', return_value=backend), \
                patch('builtins.input', side_effect=lambda *args: next(inputs)), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main([
                '--sample', '--mode', 'retriever-reader',
                '--history-mode', 'concat', '--history-turns', '1',
            ]), 0)

        second = backend.answer.call_args_list[1]
        self.assertEqual(second.args[0], 'Sốt xuất huyết lây thế nào?')
        self.assertIn('Câu hỏi trước: Sốt xuất huyết do virus nào?',
                      second.kwargs['reader_question'])
        self.assertIn('Câu hỏi hiện tại: Sốt xuất huyết lây thế nào?',
                      second.kwargs['reader_question'])
