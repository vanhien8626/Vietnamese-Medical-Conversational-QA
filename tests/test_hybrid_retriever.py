import unittest

from hybrid_retriever import HybridDocumentIndex, SemanticEncoderError


class FakeEncoder:
    model_name = 'fake-semantic'

    def encode(self, texts):
        vectors = {
            'Người bệnh cần nghỉ ngơi.': [1.0, 0.0],
            'Tăng huyết áp còn được gọi là huyết áp cao.': [0.0, 1.0],
            'Áp lực máu cao là bệnh gì?': [0.0, 1.0],
        }
        return [vectors.get(text, [0.5, 0.5]) for text in texts]


class HybridRetrieverTests(unittest.TestCase):
    def test_semantic_signal_finds_paraphrase_without_shared_terms(self):
        index = HybridDocumentIndex(
            'Người bệnh cần nghỉ ngơi. Tăng huyết áp còn được gọi là huyết áp cao.',
            encoder=FakeEncoder(),
            semantic_weight=0.8,
        )

        result = index.search('Áp lực máu cao là bệnh gì?', top_k=1)

        self.assertEqual(
            result[0]['text'],
            'Tăng huyết áp còn được gọi là huyết áp cao.',
        )
        self.assertGreater(result[0]['semantic_score'], 0.9)
        self.assertGreaterEqual(result[0]['candidate_count'], 1)

    def test_invalid_weight_and_vector_shape_fail_cleanly(self):
        for weight in (-0.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                HybridDocumentIndex('Tài liệu.', encoder=FakeEncoder(),
                                    semantic_weight=weight)
        with self.assertRaises(ValueError):
            HybridDocumentIndex('Tài liệu.', encoder=FakeEncoder(),
                                candidate_multiplier=0)

        broken = FakeEncoder()
        broken.encode = lambda texts: [[1.0]] * (len(texts) + 1)
        with self.assertRaises(SemanticEncoderError):
            HybridDocumentIndex('Tài liệu.', encoder=broken)
