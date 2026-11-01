"""Kết hợp BM25 với Sentence-BERT để truy xuất từ khóa và ngữ nghĩa."""
import math

from rag_engine import DocumentIndex, _is_non_content


DEFAULT_SEMANTIC_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


class SemanticEncoderError(RuntimeError):
    """Lỗi tải hoặc chạy mô hình Sentence-BERT."""


class SentenceTransformerEncoder:
    def __init__(self, model=DEFAULT_SEMANTIC_MODEL, *, device='cpu', local_only=False):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise SemanticEncoderError(
                'Hãy cài truy xuất ngữ nghĩa: python -m pip install -r requirements-hybrid.txt'
            ) from None
        try:
            self.model_name = model
            self._model = SentenceTransformer(
                model,
                device=device,
                local_files_only=local_only,
                trust_remote_code=False,
            )
        except Exception as exc:
            raise SemanticEncoderError(
                f'Không tải được mô hình embedding ({type(exc).__name__}). '
                'Kiểm tra tên mô hình, kết nối, bộ nhớ hoặc cache --local-only.'
            ) from None

    def encode(self, texts):
        try:
            vectors = self._model.encode(
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return vectors.tolist()
        except Exception as exc:
            raise SemanticEncoderError(
                f'Không mã hóa được văn bản ({type(exc).__name__}).'
            ) from None


def _cosine(left, right):
    if len(left) != len(right) or not left:
        raise SemanticEncoderError('Vector embedding không hợp lệ hoặc khác kích thước.')
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0
    return dot / (left_norm * right_norm)


class HybridDocumentIndex:
    """BM25 chọn ứng viên, Sentence-BERT xếp hạng lại theo ngữ nghĩa."""

    def __init__(self, document, *, semantic_model=DEFAULT_SEMANTIC_MODEL,
                 semantic_weight=0.35, candidate_multiplier=5,
                 device='cpu', local_only=False, encoder=None):
        if not math.isfinite(semantic_weight) or not 0 <= semantic_weight <= 1:
            raise ValueError('semantic_weight phải nằm trong [0, 1].')
        if type(candidate_multiplier) is not int or candidate_multiplier < 1:
            raise ValueError('candidate_multiplier phải là số nguyên dương.')
        self.lexical = DocumentIndex(document)
        self.document = self.lexical.document
        self.sentences = self.lexical.sentences
        self.semantic_weight = semantic_weight
        self.candidate_multiplier = candidate_multiplier
        self.encoder = encoder or SentenceTransformerEncoder(
            semantic_model, device=device, local_only=local_only)
        self.model = getattr(self.encoder, 'model_name', semantic_model)
        self._content_positions = [
            i for i, sentence in enumerate(self.sentences) if not _is_non_content(sentence)]
        content = [self.sentences[position] for position in self._content_positions]
        vectors = self.encoder.encode(content) if content else []
        if len(vectors) != len(content):
            raise SemanticEncoderError('Mô hình embedding trả sai số lượng vector.')
        self._vectors = dict(zip(self._content_positions, vectors))

    def search(self, question, history=None, *, threshold=0.15, top_k=3):
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError('threshold phải là số hữu hạn không âm.')
        if type(top_k) is not int or top_k < 1:
            raise ValueError('top_k phải là số nguyên dương.')
        if not question.strip() or not self._content_positions:
            return []

        candidate_count = min(
            len(self.sentences), max(top_k, top_k * self.candidate_multiplier))
        lexical_rows = self.lexical.search(
            question, history, threshold=0, top_k=candidate_count)
        if not lexical_rows:
            return []
        query_vector = self.encoder.encode([question])
        if len(query_vector) != 1:
            raise SemanticEncoderError('Mô hình embedding không trả đúng vector câu hỏi.')

        ranked = []
        for lexical_row in lexical_rows:
            position = lexical_row['sentence_index']
            vector = self._vectors[position]
            lexical_score = lexical_row['score']
            lexical_normalized = lexical_score / (1 + lexical_score)
            semantic_score = max(0, _cosine(query_vector[0], vector))
            score = (
                (1 - self.semantic_weight) * lexical_normalized
                + self.semantic_weight * semantic_score
            )
            if score >= threshold:
                ranked.append({
                    'text': self.sentences[position],
                    'score': score,
                    'bm25_score': lexical_row['bm25_score'],
                    'lexical_score': lexical_score,
                    'semantic_score': semantic_score,
                    'candidate_count': len(lexical_rows),
                    'sentence_index': position,
                })
        return sorted(ranked, key=lambda item: (-item['score'], item['sentence_index']))[:top_k]

