"""Hỏi–đáp dòng lệnh, không dùng mô hình AI hay dịch vụ bên ngoài."""
import argparse
import json
import math
import sys
from pathlib import Path

from rag_engine import DocumentIndex, FALLBACK


def main(argv=None):
    parser = argparse.ArgumentParser(description='Hỏi đáp tài liệu y tế bằng truy xuất từ khóa.')
    parser.add_argument('--document', type=Path, help='Tệp văn bản UTF-8; mặc định dùng tài liệu mẫu.')
    parser.add_argument('--question', help='Hỏi một câu rồi thoát.')
    parser.add_argument('--json', action='store_true', help='Xuất JSON khi dùng --question.')
    parser.add_argument('--threshold', type=float, default=0.15, help='Ngưỡng truy xuất không âm (mặc định 0.15).')
    parser.add_argument('--top-k', type=int, default=1, help='Số bằng chứng xếp hạng hiển thị (mặc định 1).')
    parser.add_argument('--export', type=Path, help='Lưu phiên hỏi đáp thành JSON; không ghi đè tệp có sẵn.')
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error('--top-k phải là số nguyên dương.')
    if args.export and args.export.exists():
        parser.error('Tệp xuất đã tồn tại; hãy chọn tên khác.')
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
        turns = []

        def ask(question):
            matches = index.search(question, history, threshold=args.threshold, top_k=args.top_k)
            best = matches[0] if matches else None
            result = {
                'answer': best['text'] if best else FALLBACK,
                'score': best['score'] if best else 0,
                'sources': [best['text']] if best else [],
                'candidates': matches,
            }
            turns.append({'question': question, **result})
            return result

        def export():
            if args.export:
                with args.export.open('x', encoding='utf-8') as output:
                    json.dump({'document': index.document, 'threshold': args.threshold,
                               'top_k': args.top_k, 'turns': turns}, output,
                              ensure_ascii=False, indent=2)
                    output.write('\n')

        def display(result):
            print('Trả lời:', result['answer'])
            for candidate in result['candidates']:
                print(f"  Câu {candidate['sentence_index'] + 1} · {candidate['score']:.3f}: {candidate['text']}")
            print(f"Điểm truy xuất: {result['score']:.3f} (không phải xác suất đúng)")

        if args.question is not None:
            if not args.question.strip():
                parser.error('Câu hỏi rỗng.')
            result = ask(args.question)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                display(result)
            export()
            return 0
        print('Hỏi đáp từ tài liệu · /trogiup để xem lệnh')
        while True:
            try:
                question = input('Bạn: ').strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question == '/thoat':
                break
            if question == '/trogiup':
                print('/thoat: kết thúc · /xoa: xóa phiên hiện tại · /lichsu: xem lịch sử')
                continue
            if question == '/lichsu':
                for number, turn in enumerate(turns, 1):
                    print(f"{number}. {turn['question']}\n   {turn['answer']}")
                if not turns:
                    print('Chưa có lịch sử.')
                continue
            if question == '/xoa':
                history.clear()
                turns.clear()
                print('Đã xóa lịch sử.')
                continue
            if not question:
                continue
            result = ask(question)
            display(result)
            history.append({'question': question, 'answer': result['answer']})
            history[:] = history[-10:]
        export()
        return 0
    except (OSError, UnicodeError) as exc:
        print(f'Lỗi đọc/ghi tệp: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
