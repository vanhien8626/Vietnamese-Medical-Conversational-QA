const form = document.querySelector('#qa-form');
const question = document.querySelector('#question');
const documentInput = document.querySelector('#document');
const messages = document.querySelector('#messages');
documentInput.addEventListener('input', () => {
  document.querySelector('#character-count').textContent = `${documentInput.value.length} ký tự`;
});
document.querySelectorAll('[data-question]').forEach(button => {
  button.addEventListener('click', () => {
    question.value = button.dataset.question;
    question.focus();
  });
});
form.addEventListener('submit', event => {
  if (event.submitter?.value === 'clear') return;
  if (!documentInput.value.trim() || !question.value.trim()) {
    event.preventDefault();
    (!documentInput.value.trim() ? documentInput : question).focus();
    document.querySelector('#submit-status').textContent = 'Vui lòng nhập tài liệu và câu hỏi.';
    return;
  }
  document.querySelector('#submit-status').textContent = 'Đang tìm bằng chứng…';
  form.querySelector('.send').disabled = true;
});
messages.scrollTop = messages.scrollHeight;
// Enter always asks a question, even though the clear button precedes it in the form.
question.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.isComposing) {
    event.preventDefault();
    if (!form.querySelector('.send').disabled) form.requestSubmit(form.querySelector('.send'));
  }
});
