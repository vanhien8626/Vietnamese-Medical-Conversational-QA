# MedTalk — Hỏi đáp y tế tiếng Việt

Bộ mã Python trả lời câu hỏi bằng bằng chứng trích từ tài liệu người dùng cung cấp. Hỗ trợ ngữ cảnh hội thoại và đánh giá trên dữ liệu mẫu.

## Cài đặt và chạy

```bash
python -m venv .venv
source .venv/bin/activate
python chat.py
```

Phần truy xuất và đánh giá cũ cùng bộ kiểm thử mặc định chỉ dùng thư viện chuẩn Python; không cần cài thêm gói, GPU, mô hình lớn hoặc API key.

## Phiên bản mô hình tự đọc tài liệu

`read_document.py` dùng mô hình đọc hiểu trích xuất đã huấn luyện, **không dùng quy tắc từ khóa trong `rag_engine.py` và không viết sẵn đáp án**. Bạn dán văn bản hoặc đưa tệp `.txt`, nhập câu hỏi và mô hình tìm đoạn trả lời trong văn bản. Không cần API key hay server.

Cài các gói tùy chọn trong môi trường Python (khuyến nghị Python 3.10–3.12):

```bash
python -m pip install -r requirements-reader.txt
```

Chạy rồi dán tài liệu; nhập `/xong` trên dòng riêng để chuyển sang hỏi đáp:

```bash
python read_document.py
```

Hoặc đọc tệp của bạn:

```bash
python read_document.py --document tai_lieu.txt
```

Hỏi một câu và xuất JSON gồm đáp án, bằng chứng, điểm mô hình và vị trí ký tự `start`/`end`:

```bash
python read_document.py --document tai_lieu.txt --question "Bệnh này do tác nhân nào gây ra?" --json
```

- Mặc định: `deepset/xlm-roberta-base-squad2`, một mô hình đọc hiểu dựa trên XLM-R đa ngôn ngữ, chạy CPU. Đây không phải mô hình đã fine-tune trên dữ liệu y tế tiếng Việt của dự án; cần đánh giá riêng chất lượng tiếng Việt.
- Lần đầu cần Internet để tải trọng số từ Hugging Face, dung lượng lưu trữ và RAM cho mô hình. Sau khi đã tải, dùng `--local-only` để chạy từ cache. Đổi cache bằng biến `HF_HOME` nếu cần.
- `--model TEN_HOAC_THU_MUC` chọn mô hình khác tương thích `AutoModelForQuestionAnswering` và có trọng số safetensors. Không chạy mã từ xa của mô hình.
- `--device cuda` dùng GPU khi đã cài PyTorch hỗ trợ CUDA; mặc định không cần GPU.
- Tài liệu dài được xử lý bằng cửa sổ token chồng lấp. Mỗi câu hỏi xử lý độc lập; hỏi đầy đủ đối tượng thay vì chỉ hỏi "nó là gì?".
- `--threshold 0.05` là ngưỡng điểm mặc định; có thể chỉnh trong [0, 1]. Điểm chưa hiệu chỉnh nên không phải độ tin cậy y khoa. Khi mô hình không tìm được đáp án hoặc điểm thấp, chương trình từ chối trả lời.
- Đây là **trích xuất đoạn trả lời**, không phải LLM tự viết giải thích dài; không đọc PDF/ảnh/URL trực tiếp. Bằng chứng và vị trí được đối chiếu với văn bản gốc.

Trạng thái kiểm chứng: đã chạy mô hình thật `deepset/xlm-roberta-base-squad2` trên CPU với một đoạn văn tiếng Việt mới. Mô hình trích đúng tác nhân Dengue, muỗi Aedes và nhóm triệu chứng; từ chối một câu hỏi về thủ đô Pháp không có trong tài liệu. Đây chỉ là smoke test 4 câu, chưa phải đánh giá trên tập test y tế đại diện. Đã bổ sung dependency `protobuf` mà tokenizer XLM-R cần khi chuyển sang fast tokenizer. Đã kiểm chứng CLI `--local-only` đọc từ cache và trả lời được mà không cần kết nối mạng; chế độ này giải quyết tên mô hình thành thư mục cache trước khi khởi tạo pipeline. Bộ kiểm thử mặc định gồm 22 ca vẫn đạt; các ca kiểm thử mô hình trong đó dùng phản hồi kiểm soát. Kết quả 100% của `evaluate.py` bên dưới chỉ thuộc bộ truy xuất cũ, không thuộc chế độ mô hình mới.

## Dùng từ dòng lệnh

Hỏi đáp tương tác trên tài liệu mẫu:

```bash
python chat.py
```

Nhập `/thoat` để kết thúc, `/xoa` để xóa phiên hiện tại, `/lichsu` để xem các lượt hỏi đáp hoặc `/trogiup` để xem lệnh. Lịch sử chỉ nằm trong bộ nhớ trừ khi bạn bật `--export`.

Dùng tài liệu UTF-8 của bạn; hỗ trợ tệp có BOM:

```bash
python chat.py --document tai_lieu.txt
```

Hỏi một câu và xuất kết quả JSON để xử lý tiếp:

```bash
python chat.py --document tai_lieu.txt --question "Triệu chứng thường gặp là gì?" --json
```

Xem tối đa 3 câu bằng chứng theo thứ hạng và lưu phiên hỏi đáp:

```bash
python chat.py --document tai_lieu.txt --top-k 3 --export hoi_thoai.json
```

`--top-k` hiển thị các ứng viên vượt ngưỡng, kèm số thứ tự câu và điểm; câu đứng đầu vẫn là câu trả lời. JSON có thêm `candidates`, còn `sources` chỉ chứa nguồn của câu trả lời. Không đảm bảo luôn đủ số câu yêu cầu.

`--export` ghi tài liệu, cấu hình và các lượt hỏi đáp của phiên hiện tại vào JSON khi thoát bình thường, gặp EOF hoặc Ctrl+C ở lời nhắc nhập. Tệp có sẵn sẽ không bị ghi đè. `/xoa` xóa cả ngữ cảnh và các lượt sẽ xuất. Tệp xuất chứa nội dung tài liệu nên hãy chọn nơi lưu phù hợp.

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

Chỉ mục đảo ánh xạ từ khóa sang câu, giúp chỉ chấm các câu có từ khóa hoặc từ mở rộng phù hợp thay vì quét toàn bộ tài liệu cho mỗi câu hỏi.

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

Tệp đầu vào phải là danh sách hội thoại có `context` và danh sách `turns`; mỗi lượt có `question` là chuỗi không rỗng. `answer` chấp nhận một chuỗi hoặc danh sách các đáp án chuẩn không rỗng; EM và F1 lấy điểm tốt nhất trên các đáp án này. Dataset rỗng hoặc sai cấu trúc sẽ báo lỗi. Báo cáo gồm EM/F1, tính nhất quán, số hội thoại/câu hỏi, số lượt không tìm được bằng chứng (`unanswered`), F1 từng hội thoại và từng dự đoán; lệnh từ chối ghi đè tệp có sẵn. Đường dẫn dữ liệu mẫu mặc định không phụ thuộc thư mục đang chạy.

Trên dữ liệu mẫu gồm 2 hội thoại / 5 câu hỏi: EM, F1 và conversation consistency đều đạt 100%. Tập mẫu nhỏ chưa chứng minh chất lượng trên dữ liệu thực tế. Điểm truy xuất là điểm xếp hạng, không phải xác suất đúng.

## Dữ liệu UIT-ViCoQA (tùy chọn)

Dữ liệu mẫu nằm trong `data/sample_dataset.json`. Công cụ `dataset_loader.py` hỗ trợ tải UIT-ViCoQA; cần tài khoản Hugging Face đã được cấp quyền truy cập dataset `uitnlp/vicoqa`.

```bash
pip install 'datasets>=3.0.0'
python dataset_loader.py --output data/vicoqa
```

## Cấu trúc chính

- `chat.py`: hỏi đáp dòng lệnh bằng từ khóa, hỗ trợ tương tác và JSON.
- `read_document.py`, `reader_engine.py`: chế độ mô hình đọc hiểu; cài gói tùy chọn trong `requirements-reader.txt`.
- `rag_engine.py`: chỉ mục tài liệu, chuẩn hóa Unicode và truy xuất bằng chứng.
- `evaluate.py`: đánh giá EM, F1 và tính nhất quán hội thoại.
- `tests/`: kiểm thử truy xuất, Unicode, lịch sử, dòng lệnh, dữ liệu lỗi và báo cáo.

Prototype phục vụ học tập, không thay thế tư vấn hoặc chẩn đoán của bác sĩ.
