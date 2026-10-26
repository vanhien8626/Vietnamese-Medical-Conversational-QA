import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from read_document import main
from speech import SpeechError, create_audio, play_audio


class SpeechTests(unittest.TestCase):
    def test_full_text_and_unicode_path_are_forwarded(self):
        text = 'Văn bản tiếng Việt.\n' * 1000
        async def synthesize(actual, path, voice, rate):
            self.assertEqual(actual, text)
            self.assertEqual(voice, 'vi-VN-HoaiMyNeural')
            path.write_bytes(b'controlled-test-audio')
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'giọng đọc.mp3'
            with patch('speech._synthesize', side_effect=synthesize):
                self.assertEqual(create_audio(text, output), output)
            self.assertEqual(output.read_bytes(), b'controlled-test-audio')
            self.assertEqual(list(Path(folder).glob('*.part.mp3')), [])

    def test_existing_file_is_preserved_and_not_sent(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'audio.mp3'
            output.write_bytes(b'original')
            with patch('speech._synthesize') as synth, self.assertRaises(SpeechError):
                create_audio('Văn bản', output)
            synth.assert_not_called()
            self.assertEqual(output.read_bytes(), b'original')

    def test_failed_synthesis_cleans_partial_files(self):
        async def fail(text, path, voice, rate):
            path.write_bytes(b'partial')
            raise SpeechError('Network unavailable')
        with tempfile.TemporaryDirectory() as folder:
            with patch('speech._synthesize', side_effect=fail), self.assertRaises(SpeechError):
                create_audio('Văn bản', Path(folder) / 'audio.mp3')
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_audio_only_does_not_load_qa_model(self):
        with patch('read_document.create_audio', return_value=Path('test.mp3')) as generate, \
                patch('read_document.play_audio') as play, \
                patch('read_document.ReadingModel') as reader, \
                patch('read_document.RetrieverReader') as combined, redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--sample', '--audio-only']), 0)
            self.assertGreater(len(generate.call_args.args[0]), 10000)
            play.assert_called_once()
            reader.assert_not_called()
            combined.assert_not_called()

    def test_playback_precedes_question_answer_model(self):
        events = []
        with patch('read_document.create_audio', side_effect=lambda *a, **kw: Path('test.mp3')), \
                patch('read_document.play_audio', side_effect=lambda *a: events.append('play')), \
                patch('read_document.ReadingModel', side_effect=lambda *a, **kw: events.append('model') or Mock()), \
                patch('builtins.input', return_value='/thoat'), redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--sample', '--read-aloud']), 0)
        self.assertEqual(events, ['play', 'model'])

    def test_windows_player_closes_device(self):
        backend = Mock()
        modes = iter(['playing', 'stopped'])
        commands = []
        def command(text, buffer, size, callback):
            commands.append(text)
            if text.startswith('status'):
                buffer.value = next(modes)
            return 0
        backend.mciSendStringW.side_effect = command
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'âm thanh.mp3'
            path.write_bytes(b'fixture')
            with patch('speech.sys.platform', 'win32'), \
                    patch('speech.ctypes.WinDLL', return_value=backend, create=True), \
                    patch('speech.time.sleep'):
                play_audio(path)
        self.assertTrue(commands[0].startswith('open '))
        self.assertTrue(commands[-1].startswith('close '))


if __name__ == '__main__':
    unittest.main()
