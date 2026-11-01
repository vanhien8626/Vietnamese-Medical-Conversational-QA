"""Từ chối yêu cầu chẩn đoán, kê đơn hoặc tính liều mang tính cá nhân."""
import re
import unicodedata


REFUSAL = (
    'Tôi không thể chẩn đoán, kê đơn hoặc tính liều thuốc cho một cá nhân. '
    'Hãy liên hệ bác sĩ, dược sĩ hoặc cơ sở y tế; nếu có dấu hiệu nguy hiểm, '
    'hãy tìm trợ giúp y tế ngay.'
)


def _normalized(question):
    return ' '.join(unicodedata.normalize('NFC', question).lower().split())


def medical_safety_reason(question):
    text = _normalized(question)
    dose_patterns = (
        r'\b(?:uống|dùng)\s+(?:bao nhiêu|mấy)\s+(?:viên|ml|mg|lần)',
        r'\bliều\s+(?:bao nhiêu|thế nào|như thế nào)',
        r'\btính\s+liều\b',
    )
    if any(re.search(pattern, text) for pattern in dose_patterns):
        return 'personal_dosage'
    if re.search(r'\b(?:kê|cho)\s+(?:tôi|em|con tôi|mẹ tôi|bố tôi)\s+(?:đơn|thuốc)\b', text):
        return 'personal_prescription'
    personal = re.search(r'\b(?:tôi|em|con tôi|mẹ tôi|bố tôi|người nhà tôi)\b', text)
    diagnosis = (
        re.search(r'\b(?:có phải|mắc|bị)\s+(?:bệnh\s+)?(?:gì|[\wÀ-ỹ]+)', text)
        or re.search(r'\bchẩn đoán\b', text)
        or re.search(r'\b(?:nên|phải)\s+(?:uống|dùng)\s+thuốc\s+gì\b', text)
    )
    if personal and diagnosis:
        return 'personal_diagnosis_or_treatment'
    return None


def safety_refusal(question):
    reason = medical_safety_reason(question)
    if not reason:
        return None
    return {
        'answer': REFUSAL,
        'score': 0,
        'sources': [],
        'start': None,
        'end': None,
        'mode': 'medical-safety',
        'reason': reason,
        'score_type': None,
    }

