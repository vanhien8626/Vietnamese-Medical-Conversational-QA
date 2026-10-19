"""Truy xuất câu bằng từ khóa và IDF, chỉ dùng thư viện chuẩn Python."""
import math
import re
import unicodedata
from collections import Counter

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


class DocumentIndex:
    """Lập chỉ mục một lần cho nhiều câu hỏi trên cùng tài liệu."""

    def __init__(self, document):
        self.document = normalize_text(document).strip()
        self.sentences = tuple(tach_cau(self.document))
        self.tokens = tuple(frozenset(tach_tu(sentence)) for sentence in self.sentences)
        frequencies = Counter(token for tokens in self.tokens for token in tokens)
        self._weights = {
            token: math.log(1 + (len(self.sentences) + 1) / (count + 1))
            for token, count in frequencies.items()
        }
        self._unknown_weight = math.log(1 + len(self.sentences) + 1)
        self._lower_sentences = tuple(sentence.lower() for sentence in self.sentences)

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
        previous = set()
        if history and any(_contains_phrase(question_lower, phrase) for phrase in REFERENCE_WORDS):
            previous = set(tach_tu(history[-1].get('question', '')))
        denominator = sum(self._weights.get(token, self._unknown_weight) for token in query)
        ranked = []
        for position, (sentence, tokens, lower) in enumerate(zip(
            self.sentences, self.tokens, self._lower_sentences
        )):
            overlap = tokens & query
            direct = sum(self._weights[token] for token in overlap) / denominator
            intent = min(1, len(tokens & expansion) / 2) if expansion else 0
            context = len(tokens & previous) / len(previous) if previous else 0
            phrase_bonus = 0.2 if any(_contains_phrase(lower, phrase) for phrase in active_intents) else 0
            score = 0.7 * direct + 0.5 * intent + phrase_bonus
            if overlap or intent:
                score += 0.05 * context
            if score > 0 and score >= threshold:
                ranked.append({'text': sentence, 'score': score, 'sentence_index': position})
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
