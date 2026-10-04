import re
import math
from collections import Counter

STOPWORDS = {"là","và","của","có","cho","được","những","các","một","này","đó","thì","ở","với","về","như","gì","nào","không","bị","hay","đến"}
REFERENCES = ("bệnh này","bệnh đó","thuốc này","thuốc đó","nó","trường hợp này","vậy","thế")


def tokenize(text):
    return [x for x in re.findall(r"[\wÀ-ỹ]+", text.lower(), re.UNICODE) if len(x)>1 and x not in STOPWORDS]


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def resolve_question(question, history):
    q = question.lower()
    if history and (any(x in q for x in REFERENCES) or len(tokenize(question)) <= 3):
        return history[-1].get("question", "") + " " + question
    return question


def cosine(a, b):
    ca, cb = Counter(a), Counter(b)
    common = set(ca) & set(cb)
    num = sum(ca[x]*cb[x] for x in common)
    da = math.sqrt(sum(v*v for v in ca.values()))
    db = math.sqrt(sum(v*v for v in cb.values()))
    return num/(da*db) if da and db else 0.0


def retrieve(document, question, history=None, top_k=3):
    history = history or []
    query = tokenize(resolve_question(question, history))
    ranked = []
    for sentence in split_sentences(document):
        score = cosine(query, tokenize(sentence))
        ranked.append((score, sentence))
    ranked.sort(reverse=True, key=lambda x:x[0])
    return ranked[:top_k]


def answer_question(document, question, history=None, threshold=0.12):
    if not document.strip():
        return {"answer":"Bạn cần nhập văn bản y tế trước.","score":0.0,"sources":[]}
    ranked = retrieve(document, question, history)
    if not ranked or ranked[0][0] < threshold:
        return {"answer":"Không tìm thấy thông tin phù hợp trong văn bản đã cung cấp.","score":ranked[0][0] if ranked else 0.0,"sources":[]}
    score, sentence = ranked[0]
    return {"answer":sentence,"score":score,"sources":[s for sc,s in ranked if sc >= threshold]}
