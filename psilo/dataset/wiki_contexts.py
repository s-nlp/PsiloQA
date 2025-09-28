# psilo/dataset/wiki.py
import time
from typing import Dict, List, Optional
from urllib.parse import quote

import requests
import wikipediaapi
from loguru import logger
from tqdm.auto import tqdm
from utils.constants import WIKI_API


def get_random_titles(lang: str, n: int, user_agent: Optional[str] = None) -> List[str]:
    """Вернёт n случайных заголовков статей (namespace=0)."""
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent or "PsiloQA/0.1 (+https://example.org; contact: youremail@example.org)"})
    titles: List[str] = []
    while len(titles) < n:
        need = min(500, n - len(titles))
        r = s.get(
            WIKI_API.format(lang=lang),
            params=dict(
                action="query",
                list="random",
                rnnamespace=0,
                rnlimit=need,
                format="json",
            ),
            timeout=20,
        )
        r.raise_for_status()
        titles.extend([x["title"] for x in r.json()["query"]["random"]])
    return titles[:n]


def get_wikipedia_intro(page_name: str, lang: str, user_agent: Optional[str] = None) -> str:
    formatted_page_name = page_name.replace(" ", "_")
    ua = user_agent or "PsiloQA/0.1 (+https://example.org; contact: youremail@example.org)"
    wiki = wikipediaapi.Wikipedia(user_agent=ua, language=lang)
    page = wiki.page(formatted_page_name)
    if not page.exists():
        return ""  # или можно raise Exception
    return page.summary


def _wiki_url(lang: str, title: str) -> str:
    # Каноничный путь вида /wiki/Title_with_Underscores
    return f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"


def get_random_pages(
    lang: str,
    n: int,
    min_len: int = 0,
    user_agent: Optional[str] = None,
    show_progress: bool = True,
    per_request_sleep: float = 0.0,
) -> List[Dict]:
    """
    Fetches full Wikipedia page texts.

    Returns list of dicts:
    {
        "language": ...,
        "title": ...,
        "url": ...,
        "passage": full_text
    }
    """
    titles = get_random_titles(lang, n, user_agent=user_agent)
    rows: List[Dict] = []

    ua = user_agent or "PsiloQA/0.1 (+https://example.org; contact: youremail@example.org)"
    wiki = wikipediaapi.Wikipedia(user_agent=ua, language=lang)

    iterator = tqdm(titles, desc=f"{lang}: fetching full pages", leave=False) if show_progress else titles
    kept = 0

    for t in iterator:
        try:
            page = wiki.page(t.replace(" ", "_"))
            if not page.exists():
                raise RuntimeError("page does not exist")

            text = (page.text or "").strip()
            if len(text) >= min_len:
                rows.append(
                    {
                        "language": lang,
                        "title": page.title or t,
                        "url": page.fullurl or _wiki_url(lang, t),
                        "passage": text,
                    }
                )
                kept += 1

        except Exception as e:
            logger.debug(f"[{lang}] skip '{t}': {e}")

        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(kept=kept, target=n)

        if per_request_sleep:
            time.sleep(per_request_sleep)

    if show_progress and hasattr(iterator, "close"):
        iterator.close()

    logger.success(f"[{lang}] kept {kept}/{n} full pages (min_len={min_len})")
    return rows
