"""Vietnamese text-to-speech through Edge TTS, with native Windows MP3 playback."""
import asyncio
import ctypes
import os
import sys
import time
import uuid
from pathlib import Path


class SpeechError(RuntimeError):
    pass


async def _synthesize(text, destination, voice, rate):
    try:
        import edge_tts
    except ImportError:
        raise SpeechError('Hãy cài giọng đọc: python -m pip install -r requirements-speech.txt') from None
    try:
        proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
        await edge_tts.Communicate(text, voice, rate=f'{rate:+d}%', proxy=proxy).save(str(destination))
    except Exception as exc:
        raise SpeechError(f'Không tạo được âm thanh ({type(exc).__name__}). '
                          'Giọng đọc cần Internet và kết nối dịch vụ Microsoft Edge TTS.') from None


def create_audio(text, output, *, voice='vi-VN-HoaiMyNeural', rate=0):
    """Synthesize the entire text; never replace an existing output file."""
    if not text.strip():
        raise ValueError('Văn bản đọc không được để trống.')
    if type(rate) is not int or not -50 <= rate <= 100:
        raise ValueError('Tốc độ đọc phải từ -50 đến 100 phần trăm.')
    output = Path(output)
    if output.suffix.lower() != '.mp3':
        raise ValueError('Tệp âm thanh phải có đuôi .mp3.')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the filename atomically, preserving any pre-existing user file.
    try:
        with output.open('xb'):
            pass
    except FileExistsError:
        raise SpeechError(f'Tệp đã tồn tại: {output}. Hãy chọn tên khác.') from None
    temporary = output.with_name(f'.{output.stem}-{uuid.uuid4().hex}.part.mp3')
    try:
        asyncio.run(_synthesize(text, temporary, voice, rate))
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise SpeechError('Dịch vụ không trả về âm thanh.')
        os.replace(temporary, output)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return output


def play_audio(path):
    """Play MP3 synchronously on Windows; return only when playback has finished."""
    if sys.platform != 'win32':
        raise SpeechError('Tự phát qua loa hiện hỗ trợ Windows. Dùng --no-play để lưu MP3 '
                          'rồi mở bằng trình phát âm thanh trên hệ điều hành của bạn.')
    path = Path(path).resolve()
    if not path.is_file():
        raise SpeechError(f'Không tìm thấy âm thanh: {path}')
    alias = 'medqa_' + uuid.uuid4().hex
    winmm = ctypes.WinDLL('winmm')
    winmm.mciSendStringW.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_void_p)
    winmm.mciSendStringW.restype = ctypes.c_uint
    winmm.mciGetErrorStringW.argtypes = (ctypes.c_uint, ctypes.c_wchar_p, ctypes.c_uint)

    def command(value):
        response = ctypes.create_unicode_buffer(512)
        error = winmm.mciSendStringW(value, response, len(response), None)
        if error:
            message = ctypes.create_unicode_buffer(512)
            winmm.mciGetErrorStringW(error, message, len(message))
            raise SpeechError(f'Windows không phát được MP3: {message.value}')
        return response.value

    command(f'open "{path}" type mpegvideo alias {alias}')
    try:
        command(f'play {alias}')
        while command(f'status {alias} mode') == 'playing':
            time.sleep(0.1)
    finally:
        winmm.mciSendStringW(f'close {alias}', None, 0, None)
