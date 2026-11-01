"""Truy xuất câu bằng BM25, chỉ dùng thư viện chuẩn Python."""
import math
import re
import unicodedata
from collections import Counter, defaultdict

STOP_WORDS = {
    'là', 'và', 'của', 'có', 'cho', 'được', 'những', 'các', 'một',
    'này', 'đó', 'thì', 'ở', 'với', 'về', 'như', 'gì', 'nào', 'không',
    'bao', 'nhiêu', 'hay', 'để',
}
REFERENCE_WORDS = (
    'bệnh này', 'bệnh đó', 'thuốc này', 'thuốc đó',
    'triệu chứng này', 'tình trạng này', 'nó', 'vậy', 'thế',
)
INTENTS = {
    'triệu chứng': {'triệu', 'chứng', 'biểu', 'hiện'},
    'nguy hiểm': {'nặng', 'sốc', 'biến', 'chứng', 'tử', 'vong', 'nguy', 'hiểm'},
    'nguyên nhân': {'do', 'virus', 'vi', 'khuẩn', 'nguyên', 'nhân'},
    'phòng ngừa': {'phòng', 'ngừa', 'tránh', 'tiêm', 'vắc'},
    'tránh thuốc': {'thuốc', 'dùng', 'aspirin', 'ibuprofen', 'chống', 'viêm'},
}
FALLBACK = 'Không tìm thấy thông tin phù hợp trong văn bản đã cung cấp.'


def normalize_text(text):
    """Chuẩn hóa Unicode tiếng Việt, giữ nội dung và dấu câu."""
    return unicodedata.normalize('NFC', text)


def tach_cau(van_ban):
    return [sentence.strip() for sentence in re.split(
        r'(?<=[.!?])\s+|\n+', normalize_text(van_ban)
    ) if sentence.strip()]


def tach_tu(text):
    words = re.findall(r'\w+', normalize_text(text).lower(), re.UNICODE)
    return [word for word in words if len(word) > 1 and word not in STOP_WORDS]


def _contains_phrase(text, phrase):
    # Ranh giới từ tránh nhận nhầm "nó" trong "nóng".
    return re.search(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', text) is not None


def _is_non_content(text):
    """Loại tiêu đề và dòng tài liệu tham khảo khỏi ứng viên trả lời."""
    if 'http://' in text.lower() or 'https://' in text.lower():
        return True
    letters = [character for character in text if character.isalpha()]
    return bool(letters) and len(text.split()) <= 16 and all(
        character.isupper() for character in letters)


class DocumentIndex:
    """Lập chỉ mục BM25 một lần cho nhiều câu hỏi trên cùng tài liệu."""

    def __init__(self, document, *, k1=1.5, b=0.75):
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError('k1 phải là số hữu hạn dương.')
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError('b phải nằm trong [0, 1].')
        self.document = normalize_text(document).strip()
        self.sentences = tuple(tach_cau(self.document))
        self._term_counts = tuple(Counter(tach_tu(sentence)) for sentence in self.sentences)
        self.tokens = tuple(frozenset(counts) for counts in self._term_counts)
        self._lengths = tuple(sum(counts.values()) for counts in self._term_counts)
        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0
        )
        self.k1 = k1
        self.b = b
        frequencies = Counter(token for tokens in self.tokens for token in tokens)
        self._weights = {
            token: math.log(1 + (len(self.sentences) - count + 0.5) / (count + 0.5))
            for token, count in frequencies.items()
        }
        self._unknown_weight = math.log(1 + (len(self.sentences) + 0.5) / 0.5)
        self._lower_sentences = tuple(sentence.lower() for sentence in self.sentences)
        self._postings = defaultdict(set)
        for position, tokens in enumerate(self.tokens):
            for token in tokens:
                self._postings[token].add(position)
        self._intent_positions = {
            phrase: frozenset(i for i, text in enumerate(self._lower_sentences)
                              if _contains_phrase(text, phrase))
            for phrase in INTENTS
        }

    def search(self, question, history=None, *, threshold=0.15, top_k=3):
        """Trả về tối đa top_k bằng chứng theo điểm giảm dần."""
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError('threshold phải là số hữu hạn không âm.')
        if type(top_k) is not int or top_k < 1:
            raise ValueError('top_k phải là số nguyên dương.')
        question_lower = normalize_text(question).strip().lower()
        query = set(tach_tu(question_lower))
        if not query:
            return []
        active_intents = [phrase for phrase in INTENTS if _contains_phrase(question_lower, phrase)]
        expansion = set().union(*(INTENTS[phrase] for phrase in active_intents))
        subject = query - expansion if active_intents else set()
        asks_definition = _contains_phrase(question_lower, 'là gì')
        asks_transmission_route = (
            'truyền' in query
            and any(token in query for token in ('qua', 'cách', 'đường'))
        )
        if asks_definition:
            subject = query
        previous = set()
        if history and any(_contains_phrase(question_lower, phrase) for phrase in REFERENCE_WORDS):
            previous = set(tach_tu(history[-1].get('question', '')))
        denominator = sum(self._weights.get(token, self._unknown_weight) for token in query)
        ranked = []
        # Chỉ chấm các câu có từ khóa; giữ thứ tự gốc để xử lý điểm hòa.
        candidates = set().union(*(self._postings.get(token, ()) for token in query | expansion))
        for position in sorted(candidates):
            sentence, tokens = self.sentences[position], self.tokens[position]
            if _is_non_content(sentence):
                continue
            overlap = tokens & query
            length_norm = 1 - self.b
            if self._average_length:
                length_norm += self.b * self._lengths[position] / self._average_length
            bm25 = sum(
                self._weights[token]
                * self._term_counts[position][token] * (self.k1 + 1)
                / (self._term_counts[position][token] + self.k1 * length_norm)
                for token in overlap
            )
            direct = bm25 / denominator
            intent = min(1, len(tokens & expansion) / 2) if expansion else 0
            subject_match = len(tokens & subject) / len(subject) if subject else 0
            context = len(tokens & previous) / len(previous) if previous else 0
            phrase_bonus = 0.2 if any(position in self._intent_positions[phrase] for phrase in active_intents) else 0
            list_bonus = 0.6 if (
                'triệu chứng' in active_intents
                and re.search(r'\b(?:gồm|bao gồm)\b', self._lower_sentences[position])
            ) else 0
            definition_bonus = 0.6 if (
                asks_definition and subject
                and subject.issubset(tokens)
                and _contains_phrase(self._lower_sentences[position], 'là')
            ) else 0
            transmission_bonus = 0.6 if (
                asks_transmission_route
                and {'vết', 'đốt'} <= tokens
            ) else 0
            score = (
                0.7 * direct + 0.5 * intent + phrase_bonus
                + 0.6 * subject_match + list_bonus + definition_bonus
                + transmission_bonus
            )
            if overlap or intent:
                score += 0.05 * context
            if score > 0 and score >= threshold:
                ranked.append({
                    'text': sentence,
                    'score': score,
                    'bm25_score': bm25,
                    'sentence_index': position,
                })
        return sorted(ranked, key=lambda item: item['score'], reverse=True)[:top_k]

    def answer(self, question, history=None, *, threshold=0.15):
        if not self.document:
            return {'answer': 'Bạn cần nhập văn bản y tế trước.', 'score': 0, 'sources': []}
        if not question.strip():
            return {'answer': 'Bạn cần nhập câu hỏi.', 'score': 0, 'sources': []}
        matches = self.search(question, history, threshold=threshold, top_k=1)
        if not matches:
            return {'answer': FALLBACK, 'score': 0, 'sources': []}
        best = matches[0]
        return {'answer': best['text'], 'score': best['score'], 'sources': [best['text']]}


def answer_question(document, question, history=None, threshold=0.15):
    """API tương thích: hỏi một câu mà không cần quản lý chỉ mục."""
    return DocumentIndex(document).answer(question, history, threshold=threshold)
