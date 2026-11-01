import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock, patch

from medical_safety import medical_safety_reason, safety_refusal
from read_document import main


class MedicalSafetyTests(unittest.TestCase):
    def test_personal_diagnosis_dosage_and_prescription_are_refused(self):
        cases = {
            'Tôi bị bệnh gì?': 'personal_diagnosis_or_treatment',
            'Tôi nên uống thuốc gì?': 'personal_diagnosis_or_treatment',
            'Thuốc này uống bao nhiêu viên?': 'personal_dosage',
            'Kê cho tôi đơn thuốc.': 'personal_prescription',
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(medical_safety_reason(question), expected)
                self.assertEqual(safety_refusal(question)['mode'], 'medical-safety')

    def test_general_document_questions_are_allowed(self):
        for question in (
            'Chẩn đoán viêm phổi dựa vào gì?',
            'Người nghi sốt xuất huyết cần tránh thuốc nào?',
            'Paracetamol là thuốc gì?',
        ):
            with self.subTest(question=question):
                self.assertIsNone(medical_safety_reason(question))

    def test_cli_refuses_before_loading_reader(self):
        with patch('read_document.ReadingModel') as reader, \
                redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()):
            self.assertEqual(main([
                '--sample', '--question', 'Tôi nên uống thuốc gì?'
            ]), 0)
        reader.assert_not_called()
        self.assertIn('không thể chẩn đoán, kê đơn hoặc tính liều', output.getvalue())

