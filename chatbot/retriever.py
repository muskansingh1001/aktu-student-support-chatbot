import json, pickle
import numpy as np

import config


class Retriever:
    def __init__(self):
        d = config.INDEX_DIR
        self.chunks = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        self.backend = json.loads((d / "meta.json").read_text())["backend"]
        if self.backend == "embed":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(config.EMBED_MODEL)
            self.emb = np.load(d / "emb.npy")
        else:
            with open(d / "tfidf.pkl", "rb") as f:
                self.vec, self.mat = pickle.load(f)

    def scores(self, query):
        if self.backend == "embed":
            q = self.model.encode([query], normalize_embeddings=True)
            return (self.emb @ q.T).ravel()
        qv = self.vec.transform([query])   # TF-IDF rows are L2-normalised, so dot product = cosine
        return (self.mat @ qv.T).toarray().ravel()

    def search(self, query, k=config.TOP_K, boost_topic=None):
        s = self.scores(query).copy()
        if boost_topic:   # small boost for chunks from the file matching the predicted intent
            for i, c in enumerate(self.chunks):
                if boost_topic in c["name"].lower():
                    s[i] += 0.05
        top = np.argsort(-s)[:k]
        return [{**self.chunks[i], "score": float(s[i])} for i in top]