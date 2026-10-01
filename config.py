from pathlib import Path

ROOT = Path(__file__).parent
PDF_DIR = ROOT / "data" / "pdfs"          # college PDFs (ABES) you already have
RAW_DIR = ROOT / "data" / "aktu_raw"      # pages/PDFs scraped from AKTU
INDEX_DIR = ROOT / "index"

# AKTU crawl settings
SEED_URLS = ["https://aktu.ac.in/"]
ALLOWED_DOMAIN = "aktu.ac.in"
MAX_PAGES = 150
MAX_DEPTH = 2
CRAWL_DELAY = 1.0   # seconds between requests (be polite)
KEYWORDS = ["exam", "result", "circular", "syllabus", "admission", "fee", "scholarship",
            "attendance", "calendar", "ordinance", "notice", "student", "academic", "faq",
            "hostel", "placement", "grievance", "back-paper", "carry"]

# Retrieval / RAG
CHUNK_WORDS = 180
CHUNK_OVERLAP = 40
TOP_K = 4
MIN_SCORE = 0.08          # below this the bot says it doesn't know
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "claude-sonnet-5-5"
SUPPORT_CONTACT = "the AKTU helpdesk / your college admin office (see aktu.ac.in)"
