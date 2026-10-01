"""Lightweight intent classifier (multinomial Naive Bayes, pure Python - no scikit-learn needed).
Used to route/boost retrieval and show the student which topic their question belongs to."""
import math
import re
from collections import Counter, defaultdict

TRAIN = {
    "fees": ["what is the fee for btech", "how much tuition do I pay", "fee payment last date", "hostel charges per year", "is there any late fee"],
    "admission": ["how to get admission", "uptac counselling process", "eligibility for btech admission", "documents required for admission", "lateral entry admission"],
    "examination": ["when are semester exams", "how to apply for re-evaluation", "carry over paper rules", "exam form last date", "how is cgpa calculated", "result declared"],
    "attendance": ["minimum attendance required", "attendance shortage condonation", "75 percent attendance rule", "can I sit in exam with low attendance"],
    "scholarships": ["scholarship for students", "how to apply for up scholarship", "merit scholarship eligibility", "fee waiver for poor students"],
    "placement": ["placement record", "which companies visit campus", "average package", "highest salary offered", "placement training"],
    "hostel": ["hostel facility", "hostel room types", "mess food in hostel", "hostel registration"],
    "library": ["library timings", "how many books in library", "borrow books rules", "digital library access"],
    "internship": ["summer internship rules", "industrial training report", "how to find internship", "internship credits"],
    "academic_calendar": ["academic calendar", "when does semester start", "holiday list", "session dates"],
    "contact_information": ["phone number of college", "what is the college phone number", "contact number", "email address of college", "how to contact helpdesk", "college address", "website of college", "whom to call for help"],
    "courses_and_programs": ["which courses are offered", "btech branches available", "mca programme", "number of seats in cse"],
    "transport": ["bus facility", "how to reach college", "nearest metro station"],
    "facilities": ["campus facilities", "sports ground", "wifi in campus", "canteen"],
}


def _tok(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class IntentClassifier:
    def __init__(self):
        self.counts = defaultdict(Counter)
        self.docs = Counter()
        for label, qs in TRAIN.items():
            for q in qs:
                self.counts[label].update(_tok(q))
                self.docs[label] += 1
        self.vocab = {w for c in self.counts.values() for w in c}
        self.total_docs = sum(self.docs.values())

    def predict(self, text):
        toks = [t for t in _tok(text) if t in self.vocab]
        logp = {}
        for label, c in self.counts.items():
            n = sum(c.values())
            lp = math.log(self.docs[label] / self.total_docs)
            for t in toks:
                lp += math.log((c[t] + 0.5) / (n + 0.5 * len(self.vocab)))
            logp[label] = lp
        m = max(logp.values())
        exp = {k: math.exp(v - m) for k, v in logp.items()}
        z = sum(exp.values())
        best = max(exp, key=exp.get)
        return best, (exp[best] / z if toks else 0.0)