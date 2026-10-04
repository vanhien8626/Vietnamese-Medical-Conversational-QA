import re
from flask import Flask, render_template, request, session

app = Flask(__name__)
app.secret_key = "medical-qa-project"

STOP_WORDS = {
    "la", "va", "cua", "co", "cho", "duoc", "nhung", "cac", "mot",
    "nay", "do", "thi", "o", "voi", "ve", "nhu", "gi", "nao", "khong"
}


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def words(text):
    return set(re.findall(r"\w+", normalize(text), flags=re.UNICODE))


def expand_question(question, history):
    q = normalize(question)
    references = ["bệnh này", "thuốc này", "nó", "trường hợp này", "vậy", "thế"]
    if history and (len(words(q)) <= 5 or any(x in q for x in references)):
        return history[-1]["question"] + " " + question
    return question


def find_answer(document, question, history):
    if not document.strip():
        return "Bạn cần nhập văn bản y tế trước."
    if not question.strip():
        return "Bạn cần nhập câu hỏi."

    query = words(expand_question(question, history))
    best = ""
    best_score = 0

    for sentence in split_sentences(document):
        sentence_words = words(sentence)
        overlap = len(query.intersection(sentence_words))
        score = overlap / max(1, len(query))
        if score > best_score:
            best = sentence
            best_score = score

    if best_score < 0.1:
        return "Không tìm thấy thông tin phù hợp trong văn bản đã cung cấp."
    return best


@app.route("/", methods=["GET", "POST"])
def index():
    history = session.get("history", [])
    document = session.get("document", "")

    if request.method == "POST":
        if request.form.get("action") == "clear":
            session.clear()
            return render_template("index.html", document="", history=[])

        document = request.form.get("document", "").strip()
        question = request.form.get("question", "").strip()
        session["document"] = document

        if question:
            answer = find_answer(document, question, history)
            history.append({"question": question, "answer": answer})
            session["history"] = history[-10:]

    return render_template("index.html", document=document, history=session.get("history", []))


if __name__ == "__main__":
    app.run(debug=True)
