# MedTalk — Hỏi đáp y tế tiếng Việt

Bộ mã Python trả lời câu hỏi bằng bằng chứng trích từ tài liệu người dùng cung cấp. Hỗ trợ ngữ cảnh hội thoại và đánh giá trên dữ liệu mẫu.

## Cài đặt và chạy

```bash
python -m venv .venv
source .venv/bin/activate
python evaluate.py
```

Phần truy xuất, đánh giá và kiểm thử chỉ dùng thư viện chuẩn Python; không cần cài thêm gói, GPU, mô hình lớn hoặc API key.

## Hỏi đáp bằng Python

```python
from rag_engine import answer_question

van_ban = "Người bệnh nên nghỉ ngơi và uống đủ nước."
ket_qua = answer_question(van_ban, "Người bệnh nên làm gì?")
print(ket_qua["answer"])
```

Truyền danh sách `history` gồm các mục `question` và `answer` khi cần ngữ cảnh hội thoại. Bắt đầu danh sách mới khi đổi tài liệu.

## Truy xuất và đánh giá

`rag_engine.py` tách câu, chuẩn hóa từ và loại từ dừng; xếp hạng bằng IDF, mức khớp từ khóa và mở rộng ý định câu hỏi. Lịch sử chỉ hỗ trợ xếp hạng khi có tham chiếu. Nếu điểm thấp hơn ngưỡng, hệ thống thông báo không tìm thấy thông tin phù hợp.

```bash
python -m unittest discover -s tests -v
python evaluate.py
```

Trên dữ liệu mẫu gồm 2 hội thoại / 5 câu hỏi: EM, F1 và conversation consistency đều đạt 100%. Tập mẫu nhỏ chưa chứng minh chất lượng trên dữ liệu thực tế. Điểm truy xuất là điểm xếp hạng, không phải xác suất đúng.

## Dữ liệu UIT-ViCoQA (tùy chọn)

Dữ liệu mẫu nằm trong `data/sample_dataset.json`. Công cụ `dataset_loader.py` hỗ trợ tải UIT-ViCoQA; cần tài khoản Hugging Face đã được cấp quyền truy cập dataset `uitnlp/vicoqa`.

```bash
pip install 'datasets>=3.0.0'
python dataset_loader.py --output data/vicoqa
```

## Cấu trúc chính

- `rag_engine.py`: truy xuất bằng chứng.
- `evaluate.py`: đánh giá EM, F1 và tính nhất quán hội thoại.
- `tests/test_workflow.py`: kiểm thử truy xuất.

Prototype phục vụ học tập, không thay thế tư vấn hoặc chẩn đoán của bác sĩ.
