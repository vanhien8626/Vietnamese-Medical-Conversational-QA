"""Bộ nhớ hội thoại nhỏ để giải quyết tham chiếu trong câu hỏi tiếp theo."""
import re
import unicodedata


REFERENCE_REPLACEMENTS = (
    ('triệu chứng này', 'triệu chứng của {topic}'),
    ('triệu chứng đó', 'triệu chứng của {topic}'),
    ('dấu hiệu này', 'dấu hiệu của {topic}'),
    ('dấu hiệu đó', 'dấu hiệu của {topic}'),
    ('tình trạng này', '{topic}'),
    ('tình trạng đó', '{topic}'),
    ('bệnh này', '{topic}'),
    ('bệnh đó', '{topic}'),
    ('nó', '{topic}'),
)


def _clean(text):
    return ' '.join(unicodedata.normalize('NFC', text).strip().split())


def _replace_phrase(text, phrase, replacement):
    return re.sub(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', replacement,
                  text, count=1, flags=re.IGNORECASE)


def infer_topic(question, previous=None):
    """Suy ra chủ đề ngắn từ một câu hỏi độc lập bằng quy tắc thận trọng."""
    text = _clean(question).rstrip('?.!')
    lowered = text.lower()
    if previous and any(re.search(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', lowered)
                        for phrase, _ in REFERENCE_REPLACEMENTS):
        return previous

    # Ví dụ: “Khi nào người mắc sốt xuất huyết cần đi khám?”
    match = re.search(r'\bngười (?:mắc|bị)\s+(.+?)\s+(?:cần|nên|có|phải)\b',
                      text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Ví dụ: “Sốt xuất huyết do virus nào gây ra?”
    match = re.match(r'^(.+?)\s+(?:là|do|có|lây|gây|xuất hiện|được)\b',
                     text, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        if candidate.lower() not in {'ai', 'gì', 'khi nào', 'nguyên nhân', 'triệu chứng'}:
            return candidate

    # Ví dụ: “Triệu chứng của viêm phổi gồm những gì?”
    match = re.search(
        r'^(?:nguyên nhân|triệu chứng|dấu hiệu|điều trị|phòng ngừa|biến chứng)'
        r'(?:\s+\w+){0,2}\s+của\s+(.+?)(?:\s+(?:là|gồm|bao gồm|có))?$',
        text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return previous


class ConversationMemory:
    """Lưu vài lượt gần nhất và viết lại câu hỏi có đại từ tham chiếu."""

    def __init__(self, max_turns=5):
        if type(max_turns) is not int or max_turns < 1:
            raise ValueError('max_turns phải là số nguyên dương.')
        self.max_turns = max_turns
        self.turns = []
        self.topic = None

    def resolve(self, question):
        original = _clean(question)
        if not original or not self.topic:
            return original, False
        resolved = original
        for phrase, template in REFERENCE_REPLACEMENTS:
            replacement = template.format(topic=self.topic)
            updated = _replace_phrase(resolved, phrase, replacement)
            if updated != resolved:
                return updated, True
        return original, False

    def record(self, question, resolved_question, answer):
        topic = infer_topic(resolved_question, self.topic)
        if topic:
            self.topic = topic
        self.turns.append({
            'question': _clean(question),
            'resolved_question': _clean(resolved_question),
            'answer': _clean(answer),
        })
        self.turns = self.turns[-self.max_turns:]

    def contextualize(self, question, max_history=None):
        """Nối các lượt gần nhất theo cùng định dạng dùng khi fine-tune."""
        limit = self.max_turns if max_history is None else max_history
        if type(limit) is not int or limit < 0:
            raise ValueError('max_history phải là số nguyên không âm.')
        turns = self.turns[-limit:] if limit else []
        parts = [
            f"Câu hỏi trước: {turn['question']} Trả lời trước: {turn['answer']}"
            for turn in turns
        ]
        parts.append(f'Câu hỏi hiện tại: {_clean(question)}')
        return ' '.join(parts)

    def clear(self):
        self.turns.clear()
        self.topic = None
