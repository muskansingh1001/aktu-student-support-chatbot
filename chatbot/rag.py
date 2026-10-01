import os, re

import requests

import config
from chatbot.intent import IntentClassifier
from chatbot.retriever import Retriever

SYSTEM = (
    "You are a university student-support assistant for AKTU-affiliated students. "
    "Answer ONLY from the provided context. If the context does not contain the answer, say you are "
    f"not sure and point the student to {config.SUPPORT_CONTACT}. Be concise and friendly, use short "
    "bullet points for steps, and mention that dates, fees and rules should be confirmed on the official site."
)

HF_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = os.getenv("HF_MODEL", "openai/gpt-oss-20b")


class Chatbot:
    def __init__(self):
        self.retriever = Retriever()
        self.intent = IntentClassifier()
        self.hf_token = os.getenv("HF_TOKEN")
        self.client = None
        if not self.hf_token and os.getenv("ANTHROPIC_API_KEY"):
            import anthropic
            self.client = anthropic.Anthropic()

    def _llm(self, messages):
        if self.hf_token:
            r = requests.post(
                HF_URL,
                headers={"Authorization": f"Bearer {self.hf_token}"},
                json={"model": HF_MODEL, "messages": [{"role": "system", "content": SYSTEM}] + messages, "max_tokens": 500},
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        resp = self.client.messages.create(model=config.LLM_MODEL, max_tokens=500, system=SYSTEM, messages=messages)
        return resp.content[0].text

    def ask(self, question, history=None):
        topic, conf = self.intent.predict(question)
        hits = self.retriever.search(question, boost_topic=topic if conf > 0.3 else None)
        if not hits or hits[0]["score"] < config.MIN_SCORE:
            return {"answer": f"I'm not sure about that. Please check {config.SUPPORT_CONTACT}.",
                    "sources": [], "topic": topic, "confidence": conf}
        context = "\n\n".join(f"[{i+1}] ({h['source']})\n{h['text']}" for i, h in enumerate(hits))
        if self.hf_token or self.client:
            msgs = (history or [])[-6:] + [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
            try:
                answer = self._llm(msgs)
            except Exception as e:
                answer = self._extractive(question, hits) + f"\n\n(AI service error: {type(e).__name__} - showing document text instead.)"
        else:
            answer = self._extractive(question, hits)
        return {"answer": answer, "sources": sorted({h["source"] for h in hits}), "topic": topic, "confidence": conf}

    @staticmethod
    def _extractive(question, hits):
        """No-LLM fallback: return the sentences from top chunks that best overlap the question."""
        q = set(re.findall(r"\w+", question.lower()))
        sents = []
        for h in hits:
            for s in re.split(r"(?<=[.!?])\s+|\n", h["text"]):
                w = set(re.findall(r"\w+", s.lower()))
                if len(s) > 25:
                    sents.append((len(q & w) / (len(w) ** 0.5 + 1), s.strip()))
        best = [s for _, s in sorted(sents, reverse=True)[:3]]
        return "\n".join(f"- {s}" for s in best) + "\n\n(Set HF_TOKEN or ANTHROPIC_API_KEY for fuller AI-written answers.)"