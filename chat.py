"""Hỏi–đáp dòng lệnh, không dùng mô hình AI hay dịch vụ bên ngoài."""
import argparse
import json
import math
import sys
from pathlib import Path

from rag_engine import DocumentIndex


def main(argv=None):
    parser = argparse.ArgumentParser(description='Hỏi đáp tài liệu y tế bằng truy xuất từ khóa.')
    parser.add_argument('--document', type=Path, help='Tệp văn bản UTF-8; mặc định dùng tài liệu mẫu.')
    parser.add_argument('--question', help='Hỏi một câu rồi thoát.')
    parser.add_argument('--json', action='store_true', help='Xuất JSON khi dùng --question.')
    parser.add_argument('--threshold', type=float, default=0.15, help='Ngưỡng truy xuất không âm (mặc định 0.15).')
    args = parser.parse_args(argv)
    if args.json and args.question is None:
        parser.error('--json cần đi cùng --question.')
    if not math.isfinite(args.threshold) or args.threshold < 0:
        parser.error('--threshold phải là số hữu hạn không âm.')
    try:
        if args.document:
            document = args.document.read_text(encoding='utf-8-sig')
        else:
            sample = Path(__file__).resolve().parent / 'data/sample_dataset.json'
            document = json.loads(sample.read_text(encoding='utf-8'))[0]['context']
        if not document.strip():
            parser.error('Tài liệu rỗng.')
        index = DocumentIndex(document)
        history = []

        def ask(question):
            return index.answer(question, history, threshold=args.threshold)

        def display(result):
            print('Trả lời:', result['answer'])
            for source in result['sources']:
                print('Bằng chứng:', source)
            print(f"Điểm truy xuất: {result['score']:.3f} (không phải xác suất đúng)")

        if args.question is not None:
            if not args.question.strip():
                parser.error('Câu hỏi rỗng.')
            result = ask(args.question)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                display(result)
            return 0
        print('Hỏi đáp từ tài liệu · /thoat để dừng · /xoa để xóa lịch sử')
        while True:
            try:
                question = input('Bạn: ').strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question == '/thoat':
                break
            if question == '/xoa':
                history.clear()
                print('Đã xóa lịch sử.')
                continue
            if not question:
                continue
            result = ask(question)
            display(result)
            history.append({'question': question, 'answer': result['answer']})
            history[:] = history[-10:]
        return 0
    except (OSError, UnicodeError) as exc:
        print(f'Không đọc được tài liệu: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
