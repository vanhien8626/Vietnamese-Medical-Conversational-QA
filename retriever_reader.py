"""Lexical retrieval followed by neural extractive reading of selected contexts."""
import math
import re

from rag_engine import DocumentIndex, FALLBACK
from reader_engine import DEFAULT_MODEL, NO_ANSWER, ReaderError, ReadingModel


def sentence_spans(document):
    """Return sentence bounds in the original text, including decomposed Unicode."""
    boundaries = [0]
    spans = []
    for separator in re.finditer(r'(?<=[.!?])\s+|\n+', document):
        boundaries.extend((separator.start(), separator.end()))
    boundaries.append(len(document))
    for left, right in zip(boundaries[::2], boundaries[1::2]):
        text = document[left:right]
        start = left + len(text) - len(text.lstrip())
        end = right - (len(text) - len(text.rstrip()))
        if start < end:
            spans.append((start, end))
    return spans


class RetrieverReader:
    """Index once, retrieve top-k sentences, then read merged neighboring windows.

    Offsets are always measured in the original input document, not concatenated
    evidence or Unicode-normalized text. Each question is processed independently.
    """

    def __init__(self, document, *, reader=None, model=DEFAULT_MODEL, device='cpu',
                 local_only=False, top_k=3, retrieval_threshold=0.15, context_sentences=1):
        if type(top_k) is not int or top_k < 1:
            raise ValueError('top_k phải là số nguyên dương.')
        if type(context_sentences) is not int or context_sentences < 0:
            raise ValueError('context_sentences phải là số nguyên không âm.')
        if not math.isfinite(retrieval_threshold) or retrieval_threshold < 0:
            raise ValueError('retrieval_threshold phải là số hữu hạn không âm.')
        if not document.strip():
            raise ValueError('Tài liệu không được để trống.')
        self.document = document
        self.index = DocumentIndex(document)
        self.spans = sentence_spans(document)
        if len(self.spans) != len(self.index.sentences):
            raise ValueError('Không ánh xạ được câu truy xuất sang tài liệu gốc.')
        self.reader = reader
        self.reader_options = dict(model=model, device=device, local_only=local_only)
        self.top_k = top_k
        self.retrieval_threshold = retrieval_threshold
        self.context_sentences = context_sentences

    def answer(self, question, *, threshold=0.05):
        if not question.strip():
            raise ValueError('Câu hỏi không được để trống.')
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('threshold phải nằm trong [0, 1].')
        passages = self.index.search(question, top_k=self.top_k, threshold=self.retrieval_threshold)
        result = {'answer': FALLBACK, 'score': 0, 'sources': [], 'start': None, 'end': None,
                  'mode': 'retriever-reader', 'passages': passages, 'contexts': [],
                  'model': self.reader_options['model'], 'reason': 'no_retrieval_match'}
        if not passages:
            return result  # Do not load a model or silently read the whole document.
        intervals = sorted((max(0, p['sentence_index'] - self.context_sentences),
                            min(len(self.spans) - 1, p['sentence_index'] + self.context_sentences))
                           for p in passages)
        merged = []
        for left, right in intervals:
            if merged and left <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], right)
            else:
                merged.append([left, right])
        if self.reader is None:
            self.reader = ReadingModel(**self.reader_options)
        result['model'] = self.reader.model
        best = None
        for left, right in merged:
            start, end = self.spans[left][0], self.spans[right][1]
            text = self.document[start:end]
            prediction = self.reader.answer(text, question, threshold=threshold)
            result['contexts'].append({'start': start, 'end': end, 'text': text,
                                       'reader_answer': prediction['answer'],
                                       'reader_score': prediction['score'],
                                       'answered': bool(prediction['sources'])})
            if not prediction['sources']:
                continue
            local_start, local_end = prediction['start'], prediction['end']
            if (type(local_start) is not int or type(local_end) is not int or
                    not 0 <= local_start < local_end <= len(text) or
                    text[local_start:local_end] != prediction['answer']):
                raise ReaderError('Vị trí đáp án không khớp đoạn truy xuất.')
            if best is None or prediction['score'] > best['score']:
                best = {**prediction, 'start': start + local_start,
                        'end': start + local_end, 'sources': [text]}
        if best:
            result.update({key: best[key] for key in ('answer', 'score', 'sources', 'start', 'end')})
            result['reason'] = 'answered'
        else:
            result.update(answer=NO_ANSWER, reason='reader_abstained')
        return result
