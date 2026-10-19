import json
import unittest
from pathlib import Path

from app import app
from rag_engine import answer_question
from evaluate import danh_gia_du_lieu


class RetrievalTests(unittest.TestCase):
    def test_sample_dataset(self):
        data = json.loads((Path(__file__).resolve().parents[1] / 'data/sample_dataset.json').read_text())
        self.assertEqual(danh_gia_du_lieu(data), (1, 1, 1))

    def test_unseen_document_and_question_intents(self):
        doc = ('Viêm phổi có thể do vi khuẩn gây ra. '
               'Triệu chứng của viêm phổi gồm ho và khó thở. '
               'Viêm phổi nặng có thể gây biến chứng suy hô hấp. '
               'Tiêm vắc xin giúp phòng ngừa viêm phổi.')
        cases = [
            ('Viêm phổi có triệu chứng gì?', 'Triệu chứng của viêm phổi gồm ho và khó thở.'),
            ('Nguyên nhân của viêm phổi là gì?', 'Viêm phổi có thể do vi khuẩn gây ra.'),
            ('Phòng ngừa viêm phổi như thế nào?', 'Tiêm vắc xin giúp phòng ngừa viêm phổi.'),
            ('Bệnh này có nguy hiểm không?', 'Viêm phổi nặng có thể gây biến chứng suy hô hấp.'),
        ]
        for question, expected in cases:
            with self.subTest(question=question):
                result = answer_question(doc, question, [{'question': 'Viêm phổi có triệu chứng gì?'}])
                self.assertEqual(result['answer'], expected)
                self.assertEqual(result['sources'], [expected])

    def test_fallback_and_empty_inputs(self):
        self.assertEqual(answer_question('', 'Ho là gì?')['sources'], [])
        self.assertEqual(answer_question('Ho và sốt.', '')['sources'], [])
        result = answer_question('Người bệnh cần nghỉ ngơi.', 'Thiên hà Andromeda?')
        self.assertEqual(result['answer'], 'Không tìm thấy thông tin phù hợp trong văn bản đã cung cấp.')
        self.assertEqual(result['sources'], [])


class WebTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True, SECRET_KEY='test-session-key')
        self.client = app.test_client()

    def test_render_assets_and_safe_output(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        for asset in ('style.css', 'app.js'):
            with self.client.get('/static/' + asset) as response:
                self.assertEqual(response.status_code, 200)
        response = self.client.post('/', data={'document': '<script>alert(1)</script>', 'question': '<script>alert(1)</script>'})
        self.assertNotIn(b'<script>alert(1)</script>', response.data)

    def test_history_reset_on_document_change_and_clear(self):
        self.client.post('/', data={'document': 'Ho là triệu chứng.', 'question': 'Ho là gì?'})
        self.client.post('/', data={'document': 'Nghỉ ngơi giúp phục hồi.', 'question': 'Nghỉ ngơi giúp gì?'})
        with self.client.session_transaction() as session:
            self.assertEqual(len(session['history']), 1)
            self.assertEqual(session['history'][0]['question'], 'Nghỉ ngơi giúp gì?')
        self.client.post('/', data={'action': 'clear'})
        with self.client.session_transaction() as session:
            self.assertEqual(session['history'], [])


if __name__ == '__main__':
    unittest.main()
