"""Đọc tài liệu và hỏi đáp bằng mô hình; không cần sửa mã hay viết sẵn đáp án."""
import argparse
import json
import math
import sys
from pathlib import Path

from reader_engine import DEFAULT_MODEL, ReaderError, ReadingModel


def main(argv=None):
    parser = argparse.ArgumentParser(description='Mô hình tự tìm đoạn đáp án trong tài liệu bạn nhập.')
    parser.add_argument('--document', type=Path, help='Đường dẫn tệp văn bản UTF-8.')
    parser.add_argument('--question', help='Hỏi một câu rồi thoát; nếu bỏ trống sẽ chat tương tác.')
    parser.add_argument('--json', action='store_true', help='Xuất JSON khi dùng --question.')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Tên trên Hugging Face hoặc thư mục mô hình.')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--local-only', action='store_true', help='Chỉ dùng mô hình đã lưu trên máy.')
    parser.add_argument('--threshold', type=float, default=0.05, help='Ngưỡng điểm mô hình trong [0, 1].')
    args = parser.parse_args(argv)
    if args.json and args.question is None:
        parser.error('--json cần đi kèm --question.')
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error('--threshold phải nằm trong [0, 1].')
    if args.question is not None and not args.question.strip():
        parser.error('Câu hỏi không được để trống.')
    try:
        if args.document:
            document = args.document.read_text(encoding='utf-8-sig')
        else:
            print('Dán tài liệu, nhập /xong trên một dòng riêng để kết thúc:', file=sys.stderr)
            lines = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                if line.strip() == '/xong':
                    break
                lines.append(line)
            document = '\n'.join(lines)
        if not document.strip():
            parser.error('Tài liệu rỗng.')
        print('Đang tải mô hình đọc hiểu…', file=sys.stderr)
        reader = ReadingModel(args.model, device=args.device, local_only=args.local_only)

        def ask(question):
            result = reader.answer(document, question, threshold=args.threshold)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print('Trả lời:', result['answer'])
                for source in result['sources']:
                    print('Bằng chứng:', source)
                print(f"Điểm mô hình: {result['score']:.4f} (chưa hiệu chỉnh độ tin cậy)")

        if args.question is not None:
            ask(args.question)
            return 0
        print('Nhập câu hỏi đầy đủ · /thoat để dừng. Mỗi câu được xử lý độc lập.')
        while True:
            try:
                question = input('Bạn: ').strip()
            except EOFError:
                break
            if question == '/thoat':
                break
            if question:
                ask(question)
        return 0
    except KeyboardInterrupt:
        print('\nĐã dừng.', file=sys.stderr)
        return 130
    except (OSError, UnicodeError, ReaderError, ValueError) as exc:
        print(f'Lỗi: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
