import os
from flask import Flask, render_template, request, session
from rag_engine import answer_question

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-in-production")

SAMPLE_TEXT = """Sốt xuất huyết là bệnh truyền nhiễm do virus Dengue gây ra và lây truyền qua muỗi Aedes. Triệu chứng thường gặp gồm sốt cao đột ngột, đau đầu, đau mỏi cơ, đau sau hốc mắt và buồn nôn. Một số trường hợp nặng có thể gây sốc, chảy máu nặng hoặc tổn thương cơ quan. Người có dấu hiệu cảnh báo cần đến cơ sở y tế để được khám và theo dõi."""

@app.route("/", methods=["GET", "POST"])
def index():
    history = session.get("history", [])
    document = session.get("document", SAMPLE_TEXT)
    if request.method == "POST":
        if request.form.get("action") == "clear":
            session["history"] = []
            return render_template("index.html", document=document, history=[])
        document = request.form.get("document", "").strip()
        question = request.form.get("question", "").strip()
        session["document"] = document
        if question:
            result = answer_question(document, question, history)
            history.append({"question":question,"answer":result["answer"],"score":round(result["score"],3),"sources":result["sources"]})
            session["history"] = history[-10:]
    return render_template("index.html", document=document, history=session.get("history", []))

if __name__ == "__main__":
    app.run(debug=True)
