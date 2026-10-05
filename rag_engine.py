import re

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


def xu_ly_cau_hoi(cau_hoi, lich_su):
    """Bổ sung câu hỏi trước khi câu hiện tại phụ thuộc ngữ cảnh."""
    if not lich_su:
        return cau_hoi

    cau_hoi_thuong = cau_hoi.lower()
    co_tham_chieu = any(cum in cau_hoi_thuong for cum in REFERENCE_WORDS)
    cau_hoi_ngan = len(tach_tu(cau_hoi)) <= 2

    if co_tham_chieu or cau_hoi_ngan:
        cau_truoc = lich_su[-1].get("question", "")
        return cau_truoc + " " + cau_hoi

    return cau_hoi


def tinh_diem(cau, cau_hoi):
    """Tính điểm dựa trên số từ chung và độ chính xác của từ khóa."""
    tu_cau = set(tach_tu(cau))
    tu_hoi = set(tach_tu(cau_hoi))

    if not tu_cau or not tu_hoi:
        return 0

    so_tu_chung = len(tu_cau.intersection(tu_hoi))

    # Câu trả lời phải chứa được nhiều từ của câu hỏi,
    # đồng thời tránh ưu tiên câu quá dài chỉ vì có một vài từ chung.
    recall = so_tu_chung / len(tu_hoi)
    precision = so_tu_chung / len(tu_cau)

    return 0.7 * recall + 0.3 * precision


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
    cau_hoi_day_du = xu_ly_cau_hoi(question, history)

    cau_tot_nhat = ""
    diem_cao_nhat = 0

    for cau in cac_cau:
        diem = tinh_diem(cau, cau_hoi_day_du)
        if diem > diem_cao_nhat:
            diem_cao_nhat = diem
            cau_tot_nhat = cau

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
