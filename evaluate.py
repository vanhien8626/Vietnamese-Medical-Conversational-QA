import re
from collections import Counter
from rag_engine import answer_question


def chuan_hoa(text):
    """Chuẩn hóa câu để so sánh kết quả."""
    text = text.lower().strip()
    text = re.sub(r"[^\wÀ-ỹ\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def exact_match(cau_du_doan, cau_dung):
    """EM = 1 nếu hai câu giống nhau hoàn toàn, ngược lại = 0."""
    if chuan_hoa(cau_du_doan) == chuan_hoa(cau_dung):
        return 1
    return 0


def f1_score(cau_du_doan, cau_dung):
    """Tính F1 dựa trên số từ giống nhau giữa hai câu."""
    tu_du_doan = chuan_hoa(cau_du_doan).split()
    tu_dung = chuan_hoa(cau_dung).split()

    if len(tu_du_doan) == 0 or len(tu_dung) == 0:
        return 0

    tu_chung = Counter(tu_du_doan) & Counter(tu_dung)
    so_tu_chung = sum(tu_chung.values())

    if so_tu_chung == 0:
        return 0

    precision = so_tu_chung / len(tu_du_doan)
    recall = so_tu_chung / len(tu_dung)

    return 2 * precision * recall / (precision + recall)


def danh_gia_du_lieu(du_lieu):
    tong_em = 0
    tong_f1 = 0
    tong_cau_hoi = 0
    hoi_thoai_dung = 0

    for hoi_thoai in du_lieu:
        lich_su = []
        hoi_thoai_on_dinh = True

        for luot in hoi_thoai["turns"]:
            ket_qua = answer_question(
                hoi_thoai["context"],
                luot["question"],
                lich_su
            )

            cau_du_doan = ket_qua["answer"]
            cau_dung = luot["answer"]

            em = exact_match(cau_du_doan, cau_dung)
            f1 = f1_score(cau_du_doan, cau_dung)

            tong_em += em
            tong_f1 += f1
            tong_cau_hoi += 1

            if f1 < 0.5:
                hoi_thoai_on_dinh = False

            lich_su.append({
                "question": luot["question"],
                "answer": cau_du_doan
            })

        if hoi_thoai_on_dinh:
            hoi_thoai_dung += 1

    if tong_cau_hoi == 0:
        return 0, 0, 0

    em_trung_binh = tong_em / tong_cau_hoi
    f1_trung_binh = tong_f1 / tong_cau_hoi
    consistency = hoi_thoai_dung / len(du_lieu)

    return em_trung_binh, f1_trung_binh, consistency


if __name__ == "__main__":
    import json

    with open("data/sample_dataset.json", "r", encoding="utf-8") as file:
        du_lieu = json.load(file)

    em, f1, consistency = danh_gia_du_lieu(du_lieu)

    print("===== KẾT QUẢ ĐÁNH GIÁ =====")
    print(f"Exact Match: {em:.2%}")
    print(f"F1 Score: {f1:.2%}")
    print(f"Conversation Consistency: {consistency:.2%}")
