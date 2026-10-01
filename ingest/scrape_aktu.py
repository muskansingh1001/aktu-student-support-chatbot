"""Polite, robots.txt-aware crawler for aktu.ac.in. Saves page text and PDFs to data/aktu_raw/.
Run:  python -m ingest.scrape_aktu
"""
import hashlib, time
from collections import deque
from urllib import robotparser
from urllib.parse import urljoin, urldefrag, urlparse

import requests
from bs4 import BeautifulSoup

import config

HEADERS = {"User-Agent": "UniversityStudentSupportBot/1.0 (educational project)"}


class Robots:
    """robots.txt rules, fetched with requests (urllib failed on some sites)."""

    def __init__(self):
        self.rp = None  # None means "no rules found" -> allowed

    def load(self):
        url = f"https://{config.ALLOWED_DOMAIN}/robots.txt"
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
        except Exception as e:
            print(f"Could not reach {url}: {type(e).__name__}: {e}")
            return False
        if r.status_code == 200:
            self.rp = robotparser.RobotFileParser()
            self.rp.parse(r.text.splitlines())
            print("robots.txt loaded")
        elif r.status_code >= 500:
            print(f"robots.txt gave server error {r.status_code}; try again later")
            return False
        else:
            print(f"robots.txt not found (status {r.status_code}); no crawl rules to follow")
        return True

    def can(self, url):
        return True if self.rp is None else self.rp.can_fetch(HEADERS["User-Agent"], url)


def relevant(url):
    u = url.lower()
    return u.endswith(".pdf") or any(k in u for k in config.KEYWORDS)


def save(name, data, src, ext):
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(src.encode()).hexdigest()[:10]
    path = config.RAW_DIR / f"{name}_{h}.{ext}"
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")
    (config.RAW_DIR / (path.name + ".src")).write_text(src, encoding="utf-8")


def main():
    robots = Robots()
    if not robots.load():
        print("Stopping. Check your internet connection and that https://aktu.ac.in opens in your browser.")
        return
    queue = deque((u, 0) for u in config.SEED_URLS)
    seen, pages = set(), 0
    while queue and pages < config.MAX_PAGES:
        url, depth = queue.popleft()
        url, _ = urldefrag(url)
        if url in seen or urlparse(url).netloc.replace("www.", "") != config.ALLOWED_DOMAIN:
            continue
        seen.add(url)
        if not robots.can(url):
            print("blocked by robots.txt:", url)
            continue
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
        except Exception as e:
            print("skip", url, f"{type(e).__name__}: {e}")
            continue
        time.sleep(config.CRAWL_DELAY)
        ctype = r.headers.get("content-type", "")
        slug = urlparse(url).path.strip("/").replace("/", "_")[:40] or "home"
        if url.lower().endswith(".pdf") or "pdf" in ctype:
            if len(r.content) < 15_000_000:
                save(slug, r.content, url, "pdf")
                pages += 1
                print("pdf ", url)
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "nav", "footer", "header"]):
            t.decompose()
        text = "\n".join(l.strip() for l in soup.get_text("\n").splitlines() if l.strip())
        if len(text) > 200:
            save(slug, text, url, "txt")
            pages += 1
            print("page", url)
        else:
            print("too little text:", url)
        if depth < config.MAX_DEPTH:
            for a in soup.find_all("a", href=True):
                nxt = urljoin(url, a["href"])
                if relevant(nxt):
                    queue.append((nxt, depth + 1))
    print(f"done: {pages} documents saved to {config.RAW_DIR}")


if __name__ == "__main__":
    main()