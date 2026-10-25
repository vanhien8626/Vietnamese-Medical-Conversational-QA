import unicodedata
import unittest
from unittest.mock import Mock, patch

from reader_engine import NO_ANSWER
from retriever_reader import RetrieverReader, sentence_spans


def reader_for(span):
    reader = Mock(model='controlled-reader')
    def answer(document, question, threshold=0.05):
        start = document.index(span)
        return {'answer': span, 'score': 0.8, 'sources': [document],
                'start': start, 'end': start + len(span), 'model': reader.model}
    reader.answer.side_effect = answer
    return reader


class RetrieverReaderTests(unittest.TestCase):
    def test_retrieves_before_reading_and_maps_original_offsets(self):
        document = '\n  Dinh dưỡng cần cân bằng.\n\nSốt xuất huyết do virus Dengue gây ra.\nNghỉ ngơi giúp hồi phục.  '
        reader = reader_for('Dengue')
        engine = RetrieverReader(document, reader=reader, top_k=1, context_sentences=0)
        result = engine.answer('Sốt xuất huyết do virus nào gây ra?')
        context = reader.answer.call_args.args[0]
        self.assertEqual(context, 'Sốt xuất huyết do virus Dengue gây ra.')
        self.assertNotIn('Dinh dưỡng', context)
        self.assertEqual(document[result['start']:result['end']], 'Dengue')
        self.assertEqual(result['reason'], 'answered')

    def test_decomposed_unicode_offsets_are_not_normalized(self):
        document = unicodedata.normalize('NFD', 'Đọc tài liệu. Sốt xuất huyết do virus Dengue gây ra.')
        result = RetrieverReader(document, reader=reader_for('Dengue'), context_sentences=0).answer('Virus gây sốt xuất huyết?')
        self.assertEqual(result['start'], document.index('Dengue'))
        self.assertEqual(document[result['start']:result['end']], result['answer'])

    def test_overlapping_windows_are_read_once(self):
        document = 'Sốt cao là triệu chứng. Đau đầu là triệu chứng. Buồn nôn là triệu chứng.'
        reader = reader_for('Đau đầu')
        engine = RetrieverReader(document, reader=reader, top_k=3, context_sentences=1)
        result = engine.answer('Các triệu chứng là gì?')
        self.assertEqual(len(result['contexts']), 1)
        reader.answer.assert_called_once()
        self.assertEqual(reader.answer.call_args.args[0], document)

    def test_no_match_never_loads_or_calls_reader(self):
        with patch('retriever_reader.ReadingModel') as loader:
            engine = RetrieverReader('Người bệnh nên nghỉ ngơi.')
            result = engine.answer('Thiên hà Andromeda?')
            self.assertEqual(result['reason'], 'no_retrieval_match')
            self.assertEqual(result['sources'], [])
            loader.assert_not_called()

    def test_model_can_abstain_despite_retrieval_match(self):
        reader = Mock(model='controlled-reader')
        reader.answer.return_value = {'answer': NO_ANSWER, 'score': 0.95, 'sources': [], 'start': None, 'end': None}
        result = RetrieverReader('Bệnh gây sốt.', reader=reader).answer('Bệnh gây gì?')
        self.assertEqual(result['reason'], 'reader_abstained')
        self.assertEqual(result['sources'], [])
        self.assertIsNone(result['start'])

    def test_local_reader_scores_choose_best_context_and_reuse_model(self):
        reader = Mock(model='controlled-reader')
        def answer(document, question, threshold=0.05):
            start = document.index('sốt')
            return {'answer': 'sốt', 'score': 0.9 if 'cao' in document else 0.6,
                    'sources': [document], 'start': start, 'end': start + 3}
        reader.answer.side_effect = answer
        document = 'Bệnh gây sốt nhẹ. Nghỉ ngơi giúp phục hồi. Bệnh gây sốt cao.'
        with patch('retriever_reader.ReadingModel', return_value=reader) as loader:
            engine = RetrieverReader(document, context_sentences=0, top_k=2)
            for _ in range(2):
                result = engine.answer('Bệnh gây sốt?')
                self.assertEqual(result['sources'], ['Bệnh gây sốt cao.'])
                self.assertEqual(result['start'], document.rindex('sốt'))
            loader.assert_called_once()

    def test_spans_and_parameter_validation(self):
        document = '  A.\n\n  B!  C?  '
        self.assertEqual([document[a:b] for a, b in sentence_spans(document)], ['A.', 'B!', 'C?'])
        for options in ({'top_k': 0}, {'context_sentences': -1}, {'retrieval_threshold': float('nan')}):
            with self.assertRaises(ValueError):
                RetrieverReader('Một tài liệu.', **options)


if __name__ == '__main__':
    unittest.main()
