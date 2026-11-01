"""Đọc hiểu trích xuất bằng mô hình đã huấn luyện, không dùng luật từ khóa."""
import math
from pathlib import Path

DEFAULT_MODEL = 'deepset/xlm-roberta-base-squad2'
NO_ANSWER = 'Chưa tìm được câu trả lời đủ tin cậy trong tài liệu.'


class ReaderError(RuntimeError):
    """Lỗi tải mô hình hoặc suy luận."""


class ReadingModel:
    """Tải mô hình một lần; tìm đoạn đáp án trên toàn tài liệu qua cửa sổ token."""

    def __init__(self, model=DEFAULT_MODEL, *, device='cpu', local_only=False,
                 no_answer_margin=0.0):
        if device not in ('cpu', 'cuda'):
            raise ValueError('device phải là cpu hoặc cuda.')
        if not math.isfinite(no_answer_margin) or no_answer_margin < 0:
            raise ValueError('no_answer_margin phải là số hữu hạn không âm.')
        self.model = model
        self.no_answer_margin = no_answer_margin
        try:
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
        except ImportError:
            raise ReaderError('Hãy cài gói đọc hiểu: python -m pip install -r requirements-reader.txt') from None
        try:
            source = model
            if local_only and not Path(model).is_dir():
                # Resolve the cache first so downstream pipeline helpers receive a
                # filesystem path instead of attempting remote metadata requests.
                from huggingface_hub import snapshot_download
                source = snapshot_download(model, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, local_files_only=local_only,
                                                       trust_remote_code=False)
            if not tokenizer.is_fast:
                raise ReaderError('Cần tokenizer fast để xác định đúng vị trí đáp án.')
            network = AutoModelForQuestionAnswering.from_pretrained(
                source, local_files_only=local_only, trust_remote_code=False, use_safetensors=True)
            self._qa = pipeline('question-answering', model=network, tokenizer=tokenizer,
                                device=0 if device == 'cuda' else -1)
        except ReaderError:
            raise
        except Exception as exc:
            raise ReaderError(f'Không tải được mô hình ({type(exc).__name__}). Kiểm tra tên mô hình, '
                              'kết nối Hugging Face, bộ nhớ và thiết bị. Nếu dùng --local-only, '
                              'hãy tải mô hình trước hoặc chỉ định thư mục mô hình cục bộ.') from None

    def answer(self, document, question, *, threshold=0.20):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('threshold phải nằm trong [0, 1].')
        if not document.strip() or not question.strip():
            raise ValueError('Tài liệu và câu hỏi không được để trống.')
        try:
            raw_result = self._qa(question=question.strip(), context=document, top_k=2,
                              handle_impossible_answer=True, max_seq_len=384,
                              doc_stride=128, max_question_len=64, max_answer_len=128)
        except Exception as exc:
            raise ReaderError(f'Không chạy được mô hình ({type(exc).__name__}). '
                              'Kiểm tra bộ nhớ hoặc thử --device cpu.') from None
        try:
            candidates = raw_result if isinstance(raw_result, list) else [raw_result]
            if not candidates:
                raise ValueError
            parsed = []
            for candidate in candidates:
                answer, score = candidate['answer'], float(candidate['score'])
                if not isinstance(answer, str) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError
                parsed.append((candidate, answer, score))
            no_answer_score = max(
                (score for _, answer, score in parsed if not answer.strip()),
                default=0,
            )
            answered = [item for item in parsed if item[1].strip()]
            if not answered:
                return {
                    'answer': NO_ANSWER, 'score': no_answer_score,
                    'no_answer_score': no_answer_score,
                    'score_margin': -no_answer_score, 'sources': [],
                    'start': None, 'end': None, 'model': self.model, 'mode': 'reader',
                }
            result, answer, score = max(answered, key=lambda item: item[2])
            margin = score - no_answer_score
            if score < threshold or margin <= getattr(self, 'no_answer_margin', 0.0):
                return {
                    'answer': NO_ANSWER, 'score': score,
                    'no_answer_score': no_answer_score,
                    'score_margin': margin, 'sources': [],
                    'start': None, 'end': None, 'model': self.model, 'mode': 'reader',
                }
            start, end = result['start'], result['end']
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(document):
                raise ValueError
            if document[start:end] != answer:
                raise ValueError
            # Some extractive pipelines include adjacent whitespace in the
            # predicted span. Keep offsets grounded in the original document,
            # but return the human-readable answer without those separators.
            left_padding = len(answer) - len(answer.lstrip())
            right_padding = len(answer) - len(answer.rstrip())
            start += left_padding
            end -= right_padding
            answer = document[start:end]
            if not answer:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise ReaderError('Mô hình trả kết quả không hợp lệ hoặc vị trí đáp án không khớp tài liệu.') from None
        # Bằng chứng được lấy từ tài liệu gốc, không tạo thêm nội dung.
        evidence = document[max(0, start - 120):min(len(document), end + 120)]
        return {
            'answer': answer, 'score': score,
            'no_answer_score': no_answer_score, 'score_margin': margin,
            'sources': [evidence], 'start': start, 'end': end,
            'model': self.model, 'mode': 'reader',
        }
