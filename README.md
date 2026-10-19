# MedTalk — Hỏi đáp y tế tiếng Việt

Bộ mã Python trả lời câu hỏi bằng bằng chứng trích từ tài liệu người dùng cung cấp. Hỗ trợ ngữ cảnh hội thoại và đánh giá trên dữ liệu mẫu.

## Cài đặt và chạy

```bash
python -m venv .venv
source .venv/bin/activate
python chat.py
```

Phần truy xuất, đánh giá và kiểm thử chỉ dùng thư viện chuẩn Python; không cần cài thêm gói, GPU, mô hình lớn hoặc API key.

## Dùng từ dòng lệnh

Hỏi đáp tương tác trên tài liệu mẫu:

```bash
python chat.py
```

Nhập `/thoat` để kết thúc hoặc `/xoa` để xóa lịch sử. Lịch sử chỉ nằm trong bộ nhớ và không tự ghi ra đĩa.

Dùng tài liệu UTF-8 của bạn; hỗ trợ tệp có BOM:

```bash
python chat.py --document tai_lieu.txt
```

Hỏi một câu và xuất kết quả JSON để xử lý tiếp:

```bash
python chat.py --document tai_lieu.txt --question "Triệu chứng thường gặp là gì?" --json
```

Có thể đặt ngưỡng bằng `--threshold 0.2`. Ngưỡng cao hơn khiến bộ truy xuất từ chối nhiều câu hơn; điểm này không phải xác suất câu trả lời đúng. Xem tham số bằng `python chat.py --help`.

## Hỏi đáp bằng Python

```python
from rag_engine import answer_question

van_ban = "Người bệnh nên nghỉ ngơi và uống đủ nước."
ket_qua = answer_question(van_ban, "Người bệnh nên làm gì?")
print(ket_qua["answer"])
```

Truyền danh sách `history` gồm các mục `question` và `answer` khi cần ngữ cảnh hội thoại. Bắt đầu danh sách mới khi đổi tài liệu.

Với nhiều câu hỏi trên cùng tài liệu, dùng chỉ mục tái sử dụng để tránh tách câu và tính IDF nhiều lần:

```python
from rag_engine import DocumentIndex

index = DocumentIndex(van_ban)
print(index.answer("Người bệnh nên làm gì?"))
print(index.search("Nghỉ ngơi?", top_k=3))
```

`search` trả về bằng chứng theo điểm giảm dần, kèm `sentence_index` bắt đầu từ 0. Khi điểm bằng nhau, thứ tự câu trong tài liệu được giữ nguyên. Tiếng Việt được chuẩn hóa Unicode NFC trước khi truy xuất; không hỗ trợ tự động bỏ dấu hoặc sửa lỗi chính tả.

## Truy xuất và đánh giá

`rag_engine.py` tách câu, chuẩn hóa từ và loại từ dừng; xếp hạng bằng IDF, mức khớp từ khóa và mở rộng ý định câu hỏi. Lịch sử chỉ hỗ trợ xếp hạng khi có tham chiếu. Nếu điểm thấp hơn ngưỡng, hệ thống thông báo không tìm thấy thông tin phù hợp.

```bash
python -m unittest discover -s tests -v
python evaluate.py
```

Đánh giá dữ liệu riêng và lưu kết quả từng câu:

```bash
python evaluate.py --data data/sample_dataset.json --output report.json
```

Tệp đầu vào phải là danh sách hội thoại có `context` và danh sách `turns`; mỗi lượt có `question`, `answer` là chuỗi không rỗng. Dataset rỗng hoặc sai cấu trúc sẽ báo lỗi. Báo cáo gồm EM/F1, tính nhất quán, số hội thoại/câu hỏi và từng dự đoán; lệnh từ chối ghi đè tệp có sẵn. Đường dẫn dữ liệu mẫu mặc định không phụ thuộc thư mục đang chạy.

Trên dữ liệu mẫu gồm 2 hội thoại / 5 câu hỏi: EM, F1 và conversation consistency đều đạt 100%. Tập mẫu nhỏ chưa chứng minh chất lượng trên dữ liệu thực tế. Điểm truy xuất là điểm xếp hạng, không phải xác suất đúng.

## Dữ liệu UIT-ViCoQA (tùy chọn)

Dữ liệu mẫu nằm trong `data/sample_dataset.json`. Công cụ `dataset_loader.py` hỗ trợ tải UIT-ViCoQA; cần tài khoản Hugging Face đã được cấp quyền truy cập dataset `uitnlp/vicoqa`.

```bash
pip install 'datasets>=3.0.0'
python dataset_loader.py --output data/vicoqa
```

## Cấu trúc chính

- `chat.py`: hỏi đáp dòng lệnh, hỗ trợ tương tác và JSON.
- `rag_engine.py`: chỉ mục tài liệu, chuẩn hóa Unicode và truy xuất bằng chứng.
- `evaluate.py`: đánh giá EM, F1 và tính nhất quán hội thoại.
- `tests/`: kiểm thử truy xuất, Unicode, lịch sử, dòng lệnh, dữ liệu lỗi và báo cáo.

Prototype phục vụ học tập, không thay thế tư vấn hoặc chẩn đoán của bác sĩ.
