"""Retrieval + intent evaluation.  Run:  python evaluate.py"""
from chatbot.intent import IntentClassifier
from chatbot.retriever import Retriever

TESTS = [
    ("What attendance is needed to sit in exams?", "75%", "attendance"),
    ("What are the hostel fees?", "hostel", "fees"),
    ("How is admission done for B.Tech?", "UPTAC", "admission"),
    ("What is the college phone number?", "0120", "contact_information"),
    ("Which companies visit for placements?", "Amazon", "placement"),
    ("Which courses are offered?", "MCA", "courses_and_programs"),
]
r, clf = Retriever(), IntentClassifier()
hit = intent_ok = 0
for q, kw, label in TESTS:
    top = r.search(q, k=4)
    ok = any(kw.lower() in h["text"].lower() for h in top)
    pred, _ = clf.predict(q)
    hit += ok
    intent_ok += pred == label
    print(f"{'OK ' if ok else 'MISS'} retrieval | intent={pred:<20} | {q}")
print(f"\nRetrieval hit@4: {hit}/{len(TESTS)}   Intent accuracy: {intent_ok}/{len(TESTS)}")
