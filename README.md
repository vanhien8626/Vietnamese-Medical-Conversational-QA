# Đọc hiểu văn bản hội thoại cho văn bản y tế tiếng Việt

Hệ thống hỏi–đáp hội thoại trên văn bản y tế tiếng Việt, có khả năng sử dụng ngữ cảnh của các lượt hỏi trước.

## Các yêu cầu của đồ án

### 1. Chuẩn bị dữ liệu tài liệu và câu hỏi hội thoại
`data/sample_dataset.json` lưu văn bản y tế và các lượt hỏi–đáp theo từng hội thoại. Có thể thay dữ liệu mẫu bằng UIT-ViCoQA khi thực nghiệm chính thức.

### 2. Tiền xử lý thuật ngữ y tế
`rag_engine.py` chuẩn hóa chữ thường, tách từ tiếng Việt, loại từ dừng và tách văn bản thành các câu. Các từ y tế quan trọng được giữ lại khi truy xuất.

### 3. Mô hình đọc hiểu / RAG
Đồ án chọn hướng **RAG đơn giản** thay vì fine-tune mô hình lớn. Câu hỏi hiện tại được kết hợp với ngữ cảnh hội thoại; hệ thống truy xuất các câu liên quan bằng độ tương đồng cosine và chọn bằng chứng phù hợp nhất để trả lời. Cách này nhẹ, dễ chạy và dễ giải thích khi bảo vệ đồ án.

### 4. Đánh giá
`evaluate.py` hỗ trợ ba chỉ số:
- **Exact Match (EM):** câu trả lời dự đoán có trùng đáp án chuẩn hay không.
- **F1:** mức độ trùng khớp từ giữa dự đoán và đáp án chuẩn.
- **Conversation consistency:** tỷ lệ hội thoại mà các lượt trả lời duy trì mức F1 tối thiểu.

Chạy đánh giá:
```bash
python evaluate.py
```

### 5. Kiểm soát câu trả lời ngoài dữ liệu
Nếu điểm truy xuất thấp hơn ngưỡng, hệ thống trả lời: `Không tìm thấy thông tin phù hợp trong văn bản đã cung cấp.` Hệ thống không tự tạo chẩn đoán y khoa ngoài tài liệu đầu vào.

### 6. Demo hỏi–đáp
Bản Flask dùng `app.py`. Người dùng nhập văn bản, hỏi nhiều lượt và xem câu trả lời dựa trên văn bản. `index.html` ở thư mục gốc là bản demo tĩnh dành cho GitHub Pages.

## Luồng hệ thống
`Văn bản y tế + lịch sử hội thoại + câu hỏi hiện tại -> tiền xử lý -> truy xuất RAG -> kiểm tra ngưỡng -> câu trả lời + bằng chứng`

## Cấu trúc
```text
.
├── app.py
├── rag_engine.py
├── evaluate.py
├── index.html
├── requirements.txt
├── data/
│   └── sample_dataset.json
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Chạy trên máy
```bash
pip install -r requirements.txt
python app.py
```
Mở `http://127.0.0.1:5000`.

## Tham khảo
- UIT-ViCoQA: Conversational Machine Reading Comprehension for Vietnamese Healthcare Texts.
- Dataset: UIT-ViCoQA.

## Lưu ý
Đây là prototype phục vụ học tập. Phần RAG hiện là mô hình truy xuất nhẹ để minh họa đúng quy trình của đề tài, không phải mô hình đã fine-tune trong bài báo và không thay thế tư vấn hoặc chẩn đoán của bác sĩ.