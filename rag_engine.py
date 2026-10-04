import re

# Một số từ phổ biến không có nhiều ý nghĩa khi tìm câu trả lời
STOP_WORDS = {
    "là", "và", "của", "có", "cho", "được", "những", "các", "một",
    "này", "đó", "thì", "ở", "với", "về", "như", "gì", "nào", "không"
}

# Các từ thường dùng để nhắc lại nội dung ở câu hỏi trước
REFERENCE_WORDS = ["bệnh này", "bệnh đó", "thuốc này", "thuốc đó", "nó", "vậy", "thế"]


def tach_cau(van_ban):
    """Tách văn bản thành các câu đơn giản."""
    return [cau.strip() for cau in re.split(r"(?<=[.!?])\s+|\n+", van_ban) if cau.strip()]


def tach_tu(text):
    """Chuyển về chữ thường và lấy các từ quan trọng."""
    cac_tu = re.findall(r"[\wÀ-ỹ]+", text.lower(), re.UNICODE)
    return [tu for tu in cac_tu if len(tu) > 1 and tu not in STOP_WORDS]


def xu_ly_cau_hoi(cau_hoi, lich_su):
    """Ghép câu hỏi trước nếu câu hiện tại có từ tham chiếu như 'bệnh này'."""
    cau_hoi_thuong = cau_hoi.lower()

    if lich_su:
        co_tham_chieu = any(tu in cau_hoi_thuong for tu in REFERENCE_WORDS)
        cau_hoi_ngan = len(tach_tu(cau_hoi)) <= 3

        if co_tham_chieu or cau_hoi_ngan:
            cau_truoc = lich_su[-1].get("question", "")
            return cau_truoc + " " + cau_hoi

    return cau_hoi


def tinh_diem(cau, cau_hoi):
    """Tính độ giống nhau dựa trên số từ chung."""
    tu_cau = set(tach_tu(cau))
    tu_hoi = set(tach_tu(cau_hoi))

    if len(tu_hoi) == 0:
        return 0

    so_tu_chung = len(tu_cau.intersection(tu_hoi))
    return so_tu_chung / len(tu_hoi)


def answer_question(document, question, history=None, threshold=0.1):
    """Tìm câu trong văn bản phù hợp nhất với câu hỏi."""
    if history is None:
        history = []

    if not document.strip():
        return {
            "answer": "Bạn cần nhập văn bản y tế trước.",
            "score": 0,
            "sources": []
        }

    if not question.strip():
        return {
            "answer": "Bạn cần nhập câu hỏi.",
            "score": 0,
            "sources": []
        }

    cau_hoi_day_du = xu_ly_cau_hoi(question, history)
    cau_tot_nhat = ""
    diem_cao_nhat = 0

    for cau in tach_cau(document):
        diem = tinh_diem(cau, cau_hoi_day_du)

        if diem > diem_cao_nhat:
            diem_cao_nhat = diem
            cau_tot_nhat = cau

    # Không trả lời nếu câu hỏi không liên quan đến văn bản
    if diem_cao_nhat < threshold:
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
