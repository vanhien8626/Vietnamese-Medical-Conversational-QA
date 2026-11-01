"""Optional evidence-grounded answer generation through a local Ollama server."""
import json
from urllib import error, request


class GenerationError(RuntimeError):
    """Raised when the local generation service cannot return a valid answer."""


class OllamaGenerator:
    def __init__(self, model='qwen2.5:3b', url='http://127.0.0.1:11434', timeout=120):
        if not model.strip():
            raise ValueError('Tên mô hình Ollama không được để trống.')
        if not url.startswith(('http://', 'https://')):
            raise ValueError('Địa chỉ Ollama phải bắt đầu bằng http:// hoặc https://.')
        self.model = model
        self.url = url.rstrip('/')
        self.timeout = timeout

    def generate(self, question, sources, extractive_answer):
        if not sources:
            raise ValueError('Không thể sinh câu trả lời khi chưa có bằng chứng.')
        evidence = '\n\n'.join(f'[{number}] {text}' for number, text in enumerate(sources, 1))
        prompt = (
            'Bạn là hệ thống hỏi đáp tài liệu y tế tiếng Việt. Chỉ trả lời bằng thông tin '
            'có trong BẰNG CHỨNG. Không chẩn đoán, kê đơn hoặc thêm kiến thức bên ngoài. '
            'Nếu bằng chứng không đủ, trả lời đúng câu: "Chưa đủ thông tin trong tài liệu." '
            'Trả lời ngắn gọn bằng tiếng Việt và ghi số nguồn dạng [1].\n\n'
            f'CÂU HỎI: {question}\n'
            f'ĐÁP ÁN TRÍCH XUẤT: {extractive_answer}\n'
            f'BẰNG CHỨNG:\n{evidence}\n\nTRẢ LỜI:'
        )
        body = json.dumps({
            'model': self.model,
            'prompt': prompt,
            'stream': False,
            'options': {'temperature': 0},
        }).encode('utf-8')
        http_request = request.Request(
            f'{self.url}/api/generate', data=body,
            headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode('utf-8'))
        except (error.URLError, TimeoutError, OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise GenerationError(
                f'Không gọi được Ollama tại {self.url}: {exc}') from exc
        answer = result.get('response') if isinstance(result, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise GenerationError('Ollama không trả về nội dung hợp lệ.')
        return answer.strip()
