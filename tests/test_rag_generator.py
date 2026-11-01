import io
import json
import unittest
from unittest.mock import patch

from rag_generator import GenerationError, OllamaGenerator


class _Response:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.data).encode()


class OllamaGeneratorTests(unittest.TestCase):
    @patch('rag_generator.request.urlopen')
    def test_grounded_request(self, urlopen):
        urlopen.return_value = _Response({'response': 'Do virus Dengue gây ra [1].'})
        generator = OllamaGenerator(model='test-model')

        answer = generator.generate('Do virus nào?', ['Do virus Dengue gây ra.'], 'virus Dengue')

        self.assertEqual(answer, 'Do virus Dengue gây ra [1].')
        sent = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(sent['model'], 'test-model')
        self.assertEqual(sent['options']['temperature'], 0)
        self.assertIn('[1] Do virus Dengue gây ra.', sent['prompt'])

    def test_requires_evidence(self):
        with self.assertRaises(ValueError):
            OllamaGenerator().generate('Câu hỏi?', [], 'Không rõ')

    @patch('rag_generator.request.urlopen')
    def test_invalid_response(self, urlopen):
        urlopen.return_value = _Response({'response': ''})
        with self.assertRaises(GenerationError):
            OllamaGenerator().generate('Câu hỏi?', ['Nguồn.'], 'Nguồn')


if __name__ == '__main__':
    unittest.main()
