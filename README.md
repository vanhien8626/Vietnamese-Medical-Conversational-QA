# MedTalk — Hỏi đáp y tế tiếng Việt

Ứng dụng Flask trả lời câu hỏi bằng bằng chứng trích từ tài liệu người dùng cung cấp. Giao diện hỗ trợ điện thoại, gợi ý câu hỏi và xem nguồn trả lời. Đổi tài liệu sẽ bắt đầu hội thoại mới.

## Cài đặt và chạy

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Mở `http://127.0.0.1:5000`. Bản demo không cần GPU, mô hình lớn hoặc API key.

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

- `app.py`: ứng dụng Flask và lịch sử hội thoại.
- `rag_engine.py`: truy xuất bằng chứng.
- `evaluate.py`: đánh giá EM, F1 và tính nhất quán hội thoại.
- `templates/index.html`, `static/`: giao diện.
- `tests/test_workflow.py`: kiểm thử truy xuất và ứng dụng.

Prototype phục vụ học tập, không thay thế tư vấn hoặc chẩn đoán của bác sĩ.
