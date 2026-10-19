import re
import math
from collections import Counter

# Các từ phổ biến ít có ý nghĩa khi tìm câu trả lời
STOP_WORDS = {
    "là", "và", "của", "có", "cho", "được", "những", "các", "một",
    "này", "đó", "thì", "ở", "với", "về", "như", "gì", "nào", "không",
    "bao", "nhiêu", "hay", "để"
}

# Các cụm từ thường nhắc lại nội dung ở lượt trước
REFERENCE_WORDS = [
    "bệnh này", "bệnh đó", "thuốc này", "thuốc đó",
    "triệu chứng này", "tình trạng này", "nó", "vậy", "thế"
]


def tach_cau(van_ban):
    """Tách văn bản thành các câu."""
    return [
        cau.strip()
        for cau in re.split(r"(?<=[.!?])\s+|\n+", van_ban)
        if cau.strip()
    ]


def tach_tu(text):
    """Chuyển về chữ thường và lấy các từ quan trọng."""
    cac_tu = re.findall(r"[\wÀ-ỹ]+", text.lower(), re.UNICODE)
    return [tu for tu in cac_tu if len(tu) > 1 and tu not in STOP_WORDS]


def answer_question(document, question, history=None, threshold=0.15):
    """Tìm câu phù hợp nhất trong văn bản để trả lời câu hỏi."""
    if history is None:
        history = []

    document = document.strip()
    question = question.strip()

    if not document:
        return {
            "answer": "Bạn cần nhập văn bản y tế trước.",
            "score": 0,
            "sources": []
        }

    if not question:
        return {
            "answer": "Bạn cần nhập câu hỏi.",
            "score": 0,
            "sources": []
        }

    cac_cau = tach_cau(document)
    # IDF giảm trọng số tên bệnh lặp lại trong câu hỏi và ưu tiên ý định.
    tokenized = [set(tach_tu(sentence)) for sentence in cac_cau]
    frequencies = Counter(token for tokens in tokenized for token in tokens)
    query = set(tach_tu(question))
    intents = {
        "triệu chứng": {"triệu", "chứng", "biểu", "hiện"},
        "nguy hiểm": {"nặng", "sốc", "biến", "chứng", "tử", "vong", "nguy", "hiểm"},
        "nguyên nhân": {"do", "virus", "vi", "khuẩn", "nguyên", "nhân"},
        "phòng ngừa": {"phòng", "ngừa", "tránh", "tiêm", "vắc"},
    }
    expansion = set()
    for phrase, words in intents.items():
        if phrase in question.lower():
            expansion.update(words)
    previous = set(tach_tu(history[-1].get("question", ""))) if history and any(
        phrase in question.lower() for phrase in REFERENCE_WORDS
    ) else set()

    def weight(token):
        return math.log(1 + (len(cac_cau) + 1) / (frequencies[token] + 1))

    denominator = sum(weight(token) for token in query) or 1
    cau_tot_nhat = ""
    diem_cao_nhat = 0
    for sentence, tokens in zip(cac_cau, tokenized):
        overlap = tokens & query
        direct = sum(weight(token) for token in overlap) / denominator
        intent = min(1, len(tokens & expansion) / 2) if expansion else 0
        # Ngữ cảnh chỉ hỗ trợ xếp hạng; không lấn át câu hỏi hiện tại.
        context = len(tokens & previous) / len(previous) if previous else 0
        phrase_bonus = 0.2 if any(
            phrase in question.lower() and phrase in sentence.lower()
            for phrase in intents
        ) else 0
        score = 0.7 * direct + 0.5 * intent + phrase_bonus
        if overlap or intent:
            score += 0.05 * context
        if score > diem_cao_nhat:
            diem_cao_nhat = score
            cau_tot_nhat = sentence

    # Không đoán câu trả lời khi mức liên quan quá thấp.
    if not cau_tot_nhat or diem_cao_nhat < threshold:
        return {
            "answer": "Không tìm thấy thông tin phù hợp trong văn bản đã cung cấp.",
            "score": diem_cao_nhat,
            "sources": []
        }

    return {
        "answer": cau_tot_nhat,
        "score": diem_cao_nhat,
        "sources": [cau_tot_nhat]
    }
