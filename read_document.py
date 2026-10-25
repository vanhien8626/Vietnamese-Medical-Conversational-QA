"""Đọc tài liệu và hỏi đáp bằng mô hình; không cần sửa mã hay viết sẵn đáp án."""
import argparse
import json
import math
import sys
from pathlib import Path

from reader_engine import DEFAULT_MODEL, ReaderError, ReadingModel
from retriever_reader import RetrieverReader


def main(argv=None):
    parser = argparse.ArgumentParser(description='Mô hình tự tìm đoạn đáp án trong tài liệu bạn nhập.')
    parser.add_argument('--document', type=Path, help='Đường dẫn tệp văn bản UTF-8.')
    parser.add_argument('--question', help='Hỏi một câu rồi thoát; nếu bỏ trống sẽ chat tương tác.')
    parser.add_argument('--json', action='store_true', help='Xuất JSON khi dùng --question.')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Tên trên Hugging Face hoặc thư mục mô hình.')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--local-only', action='store_true', help='Chỉ dùng mô hình đã lưu trên máy.')
    parser.add_argument('--threshold', type=float, default=0.05, help='Ngưỡng điểm mô hình trong [0, 1].')
    parser.add_argument('--mode', choices=('reader', 'retriever-reader'), default='reader',
                        help='Đọc toàn tài liệu hoặc truy xuất trước rồi đọc (mặc định reader).')
    parser.add_argument('--top-k', type=int, default=3, help='Số câu truy xuất trong chế độ retriever-reader.')
    parser.add_argument('--retrieval-threshold', type=float, default=0.15)
    parser.add_argument('--context-sentences', type=int, default=1,
                        help='Số câu lân cận mỗi phía để giữ ngữ cảnh.')
    args = parser.parse_args(argv)
    if args.top_k < 1 or args.context_sentences < 0:
        parser.error('--top-k phải dương; --context-sentences phải không âm.')
    if not math.isfinite(args.retrieval_threshold) or args.retrieval_threshold < 0:
        parser.error('--retrieval-threshold phải là số hữu hạn không âm.')
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
        if args.mode == 'retriever-reader':
            reader = RetrieverReader(document, model=args.model, device=args.device,
                                     local_only=args.local_only, top_k=args.top_k,
                                     retrieval_threshold=args.retrieval_threshold,
                                     context_sentences=args.context_sentences)
            print('Đã lập chỉ mục. Mô hình sẽ tải khi tìm được đoạn liên quan.', file=sys.stderr)
        else:
            print('Đang tải mô hình đọc hiểu…', file=sys.stderr)
            reader = ReadingModel(args.model, device=args.device, local_only=args.local_only)

        def ask(question):
            if args.mode == 'retriever-reader':
                result = reader.answer(question, threshold=args.threshold)
            else:
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
