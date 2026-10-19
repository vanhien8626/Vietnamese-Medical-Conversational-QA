import re
from collections import Counter
from rag_engine import DocumentIndex, normalize_text


def chuan_hoa(text):
    """Chuẩn hóa câu để so sánh kết quả."""
    text = normalize_text(text).lower().strip()
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


def validate_dataset(data):
    """Báo lỗi dữ liệu bằng vị trí cụ thể, không coi tập rỗng là đánh giá thành công."""
    if not isinstance(data, list) or not data:
        raise ValueError('Dataset phải là danh sách hội thoại không rỗng.')
    for i, conversation in enumerate(data):
        location = f'Hội thoại {i + 1}'
        if not isinstance(conversation, dict):
            raise ValueError(f'{location}: phải là object.')
        if not isinstance(conversation.get('context'), str) or not conversation['context'].strip():
            raise ValueError(f'{location}: context phải là chuỗi không rỗng.')
        turns = conversation.get('turns')
        if not isinstance(turns, list) or not turns:
            raise ValueError(f'{location}: turns phải là danh sách không rỗng.')
        for j, turn in enumerate(turns):
            for field in ('question', 'answer'):
                if not isinstance(turn, dict) or not isinstance(turn.get(field), str) or not turn[field].strip():
                    raise ValueError(f'{location}, lượt {j + 1}: {field} phải là chuỗi không rỗng.')


def evaluate_dataset(data):
    validate_dataset(data)
    predictions = []
    consistent = 0
    for i, conversation in enumerate(data):
        index = DocumentIndex(conversation['context'])
        history = []
        scores = []
        for j, turn in enumerate(conversation['turns']):
            result = index.answer(turn['question'], history)
            em = exact_match(result['answer'], turn['answer'])
            f1 = f1_score(result['answer'], turn['answer'])
            scores.append(f1)
            predictions.append({
                'conversation_index': i, 'turn_index': j,
                'question': turn['question'], 'expected': turn['answer'],
                'prediction': result['answer'], 'sources': result['sources'],
                'retrieval_score': result['score'], 'exact_match': em, 'f1': f1,
            })
            history.append({'question': turn['question'], 'answer': result['answer']})
        consistent += all(score >= 0.5 for score in scores)
    return {
        'conversations': len(data), 'questions': len(predictions),
        'exact_match': sum(row['exact_match'] for row in predictions) / len(predictions),
        'f1': sum(row['f1'] for row in predictions) / len(predictions),
        'conversation_consistency': consistent / len(data),
        'predictions': predictions,
    }


def danh_gia_du_lieu(du_lieu):
    """Giữ định dạng trả về cũ: EM, F1, conversation consistency."""
    report = evaluate_dataset(du_lieu)
    return report['exact_match'], report['f1'], report['conversation_consistency']


def main(argv=None):
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Đánh giá truy xuất trên dữ liệu hội thoại JSON.')
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent / 'data/sample_dataset.json')
    parser.add_argument('--output', type=Path, help='Lưu JSON gồm chỉ số và từng câu dự đoán; không ghi đè tệp có sẵn.')
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.data.read_text(encoding='utf-8-sig'))
        report = evaluate_dataset(data)
        if args.output:
            with args.output.open('x', encoding='utf-8') as output:
                json.dump(report, output, ensure_ascii=False, indent=2)
                output.write('\n')
    except (OSError, ValueError, UnicodeError) as exc:
        parser.exit(1, f'Lỗi đánh giá: {exc}\n')
    print(f"Đã đánh giá {report['conversations']} hội thoại / {report['questions']} câu hỏi.")
    print(f"Exact Match: {report['exact_match']:.2%}")
    print(f"F1 Score: {report['f1']:.2%}")
    print(f"Conversation Consistency: {report['conversation_consistency']:.2%}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
