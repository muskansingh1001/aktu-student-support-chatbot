"""Load PDFs + scraped text, chunk, and build a search index.
Run:  python -m ingest.build_index
Uses sentence-transformers embeddings if installed, otherwise TF-IDF.
"""
import json, pickle

import numpy as np
from pypdf import PdfReader

import config


def read_pdf(path):
    try:
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    except Exception as e:
        print("bad pdf", path.name, e)
        return ""


def load_documents():
    docs = []
    for folder in (config.PDF_DIR, config.RAW_DIR):
        if not folder.exists():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix == ".pdf":
                text = read_pdf(p)
            elif p.suffix == ".txt":
                text = p.read_text(encoding="utf-8", errors="ignore")
            else:
                continue
            src_file = folder / (p.name + ".src")
            source = src_file.read_text().strip() if src_file.exists() else p.name
            if len(text.strip()) > 50:
                docs.append({"source": source, "name": p.stem, "text": text})
    return docs


def chunk(text, size=config.CHUNK_WORDS, overlap=config.CHUNK_OVERLAP):
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words), 1), step) if words[i:i + size]]


def main():
    docs = load_documents()
    chunks = [{"text": c, "source": d["source"], "name": d["name"]} for d in docs for c in chunk(d["text"])]
    print(f"{len(docs)} documents -> {len(chunks)} chunks")
    config.INDEX_DIR.mkdir(exist_ok=True)
    texts = [c["text"] for c in chunks]
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.EMBED_MODEL)
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        np.save(config.INDEX_DIR / "emb.npy", emb)
        backend = "embed"
    except Exception as e:
        print("embeddings unavailable, using TF-IDF:", type(e).__name__)
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        mat = vec.fit_transform(texts)
        with open(config.INDEX_DIR / "tfidf.pkl", "wb") as f:
            pickle.dump((vec, mat), f)
        backend = "tfidf"
    (config.INDEX_DIR / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    (config.INDEX_DIR / "meta.json").write_text(json.dumps({"backend": backend}))
    print("index built with backend:", backend)


if __name__ == "__main__":
    main()
