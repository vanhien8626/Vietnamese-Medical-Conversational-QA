import re
from collections import Counter
from rag_engine import answer_question


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^\wÀ-ỹ\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def exact_match(prediction, gold):
    return float(normalize(prediction) == normalize(gold))


def f1_score(prediction, gold):
    p, g = normalize(prediction).split(), normalize(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not p or not g:
        return float(p == g)
    if same == 0:
        return 0.0
    precision, recall = same/len(p), same/len(g)
    return 2*precision*recall/(precision+recall)


def evaluate(samples):
    em = f1 = consistent = 0.0
    for sample in samples:
        history = []
        conversation_ok = True
        for turn in sample["turns"]:
            result = answer_question(sample["context"], turn["question"], history)
            pred, gold = result["answer"], turn["answer"]
            em += exact_match(pred, gold)
            f1 += f1_score(pred, gold)
            conversation_ok &= f1_score(pred, gold) >= 0.5
            history.append({"question":turn["question"],"answer":pred})
        consistent += float(conversation_ok)
    turns = sum(len(x["turns"]) for x in samples)
    return {"Exact Match": em/turns if turns else 0, "F1": f1/turns if turns else 0, "Conversation consistency": consistent/len(samples) if samples else 0}


if __name__ == "__main__":
    import json
    with open("data/sample_dataset.json", encoding="utf-8") as f:
        data=json.load(f)
    for name,value in evaluate(data).items():
        print(f"{name}: {value:.2%}")
