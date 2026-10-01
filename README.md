# AI-Powered University Student Support Chatbot (AKTU)

A Retrieval-Augmented Generation (RAG) chatbot that answers student questions about exams,
fees, attendance, scholarships, admissions, hostel, placements and more, using documents from
AKTU (aktu.ac.in) and your college PDFs (ABES data included in `data/pdfs/`).

## How it works
1. **Ingest** - `ingest/scrape_aktu.py` crawls aktu.ac.in (respects robots.txt, 1s delay) and saves pages/PDFs.
2. **Index** - `ingest/build_index.py` extracts text, splits into overlapping chunks, and builds a
   semantic index (sentence-transformers) or TF-IDF fallback.
3. **ML intent classifier** - `chatbot/intent.py` (TF-IDF + Logistic Regression) detects the topic of a
   question and boosts matching documents.
4. **RAG answer** - `chatbot/rag.py` retrieves top chunks and asks Claude to answer only from them, with
   sources. Without an API key it falls back to extractive answers. Low-confidence questions get a
   "contact the helpdesk" reply instead of a guess.
5. **UI** - Streamlit chat (`app.py`) or terminal (`chat_cli.py`).

## Run
```bash
pip install -r requirements.txt
python -m ingest.scrape_aktu        # optional: needs internet; fetches AKTU pages/PDFs
python -m ingest.build_index        # rebuild whenever data changes
export ANTHROPIC_API_KEY=...        # optional, for AI-written answers
streamlit run app.py                # or: python chat_cli.py
python evaluate.py                  # retrieval + intent check
```

## Improving it
- Add more AKTU PDFs (ordinances, syllabus, circulars) to `data/pdfs/` and re-run the index.
- Extend `TRAIN` in `chatbot/intent.py` with real student questions.
- Grow `evaluate.py` with a larger question set before changing models.
- Re-scrape regularly: circulars, dates and fees change every session.

## Notes
- The scraper targets keyword-matched pages only; check AKTU's terms of use and robots.txt before large crawls.
- Bundled ABES PDFs were compiled from public portals and may contain outdated figures; verify on official sites.
### apllication snapshots
<img width="1920" height="1080" alt="Screenshot (414)" src="https://github.com/user-attachments/assets/87caadf9-c1a2-409f-9d63-f97e6a08f508" />

