"""Fine-tune XLM-RoBERTa cho hỏi đáp trích xuất tiếng Việt."""
import argparse
import json
from pathlib import Path

from reader_engine import DEFAULT_MODEL


def load_rows(path):
    rows = []
    with Path(path).open(encoding='utf-8-sig') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Dòng {line_number}: JSON không hợp lệ.') from exc
            if not isinstance(row, dict):
                raise ValueError(f'Dòng {line_number}: mỗi mẫu phải là object.')
            for field in ('question', 'context', 'answers'):
                if field not in row:
                    raise ValueError(f'Dòng {line_number}: thiếu trường {field}.')
            if not isinstance(row['question'], str) or not row['question'].strip():
                raise ValueError(f'Dòng {line_number}: question phải là chuỗi không rỗng.')
            if not isinstance(row['context'], str) or not row['context'].strip():
                raise ValueError(f'Dòng {line_number}: context phải là chuỗi không rỗng.')
            answers = row['answers']
            if not isinstance(answers, dict):
                raise ValueError(f'Dòng {line_number}: answers phải là object.')
            texts, starts = answers.get('text'), answers.get('answer_start')
            if not isinstance(texts, list) or not isinstance(starts, list) or len(texts) != len(starts):
                raise ValueError(
                    f'Dòng {line_number}: answers.text và answers.answer_start phải là hai danh sách cùng độ dài.')
            if texts:
                if not isinstance(texts[0], str) or type(starts[0]) is not int:
                    raise ValueError(f'Dòng {line_number}: đáp án đầu tiên hoặc vị trí không hợp lệ.')
                start, text = starts[0], texts[0]
                if start < 0 or row['context'][start:start + len(text)] != text:
                    raise ValueError(f'Dòng {line_number}: answer_start không khớp context.')
            history = row.get('history', [])
            if not isinstance(history, list):
                raise ValueError(f'Dòng {line_number}: history phải là danh sách.')
            rows.append(row)
    if not rows:
        raise ValueError('Tệp dữ liệu không có mẫu hợp lệ.')
    return rows


def question_with_history(row, max_history=5):
    if type(max_history) is not int or max_history < 0:
        raise ValueError('max_history phải là số nguyên không âm.')
    history = row.get('history', [])[-max_history:] if max_history else []
    parts = []
    for turn in history:
        if not isinstance(turn, dict):
            raise ValueError('Mỗi lượt history phải là object.')
        question, answer = turn.get('question'), turn.get('answer')
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError('History cần question và answer dạng chuỗi.')
        parts.append(f'Câu hỏi trước: {question} Trả lời trước: {answer}')
    parts.append(f'Câu hỏi hiện tại: {row["question"]}')
    return ' '.join(parts)


def tokenize_rows(rows, tokenizer, *, max_length=384, doc_stride=128, max_history=5):
    questions = [question_with_history(row, max_history) for row in rows]
    contexts = [row['context'] for row in rows]
    encoded = tokenizer(
        questions,
        contexts,
        truncation='only_second',
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length',
    )
    sample_map = encoded.pop('overflow_to_sample_mapping')
    offsets = encoded.pop('offset_mapping')
    starts, ends = [], []
    for feature_index, offset in enumerate(offsets):
        sample_index = sample_map[feature_index]
        answers = rows[sample_index]['answers']
        sequence_ids = encoded.sequence_ids(feature_index)
        cls_index = encoded['input_ids'][feature_index].index(tokenizer.cls_token_id)
        if not answers['text']:
            starts.append(cls_index)
            ends.append(cls_index)
            continue
        answer_start = answers['answer_start'][0]
        answer_end = answer_start + len(answers['text'][0])
        context_tokens = [i for i, sequence in enumerate(sequence_ids) if sequence == 1]
        if not context_tokens:
            starts.append(cls_index)
            ends.append(cls_index)
            continue
        first, last = context_tokens[0], context_tokens[-1]
        if offset[first][0] > answer_start or offset[last][1] < answer_end:
            starts.append(cls_index)
            ends.append(cls_index)
            continue
        while first <= last and offset[first][0] <= answer_start:
            first += 1
        while last >= 0 and offset[last][1] >= answer_end:
            last -= 1
        starts.append(first - 1)
        ends.append(last + 1)
    encoded['start_positions'] = starts
    encoded['end_positions'] = ends
    return encoded


class FeatureDataset:
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features['input_ids'])

    def __getitem__(self, index):
        return {key: values[index] for key, values in self.features.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Fine-tune XLM-RoBERTa cho extractive QA.')
    parser.add_argument('--train-file', type=Path, required=True, help='JSONL dữ liệu train.')
    parser.add_argument('--validation-file', type=Path, required=True, help='JSONL dữ liệu validation.')
    parser.add_argument('--output', type=Path, required=True, help='Thư mục lưu checkpoint.')
    parser.add_argument('--base-model', default=DEFAULT_MODEL)
    parser.add_argument('--max-history', type=int, default=5)
    parser.add_argument('--max-length', type=int, default=384)
    parser.add_argument('--doc-stride', type=int, default=128)
    parser.add_argument('--epochs', type=float, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--local-only', action='store_true')
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Thư mục output đã có dữ liệu; hãy chọn thư mục mới.')
    if args.max_history < 0 or args.max_length < 64 or not 0 <= args.doc_stride < args.max_length:
        parser.error('max-history/max-length/doc-stride không hợp lệ.')
    if args.epochs <= 0 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error('epochs, batch-size và learning-rate phải dương.')

    try:
        from transformers import (
            AutoModelForQuestionAnswering,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            default_data_collator,
        )
    except ImportError:
        parser.error('Hãy cài dependency: python -m pip install -r requirements-training.txt')
    try:
        train_rows = load_rows(args.train_file)
        validation_rows = load_rows(args.validation_file)
        model_source = args.base_model
        if args.local_only and not Path(args.base_model).is_dir():
            from huggingface_hub import snapshot_download
            model_source = snapshot_download(args.base_model, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_source, use_fast=True, local_files_only=args.local_only,
            trust_remote_code=False)
        if not tokenizer.is_fast:
            raise ValueError('Cần tokenizer fast để ánh xạ vị trí đáp án.')
        train_data = FeatureDataset(tokenize_rows(
            train_rows, tokenizer, max_length=args.max_length,
            doc_stride=args.doc_stride, max_history=args.max_history))
        validation_data = FeatureDataset(tokenize_rows(
            validation_rows, tokenizer, max_length=args.max_length,
            doc_stride=args.doc_stride, max_history=args.max_history))
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_source, local_files_only=args.local_only,
            trust_remote_code=False, use_safetensors=True)
        training = TrainingArguments(
            output_dir=str(args.output),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            eval_strategy='epoch',
            save_strategy='epoch',
            logging_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            save_total_limit=2,
            report_to=[],
        )
        trainer = Trainer(
            model=model,
            args=training,
            train_dataset=train_data,
            eval_dataset=validation_data,
            processing_class=tokenizer,
            data_collator=default_data_collator,
        )
        trainer.train()
        trainer.save_model(str(args.output))
        tokenizer.save_pretrained(str(args.output))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'Lỗi fine-tune: {exc}\n')
    print(f'Đã lưu mô hình fine-tune: {args.output.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

