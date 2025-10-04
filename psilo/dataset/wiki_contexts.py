import asyncio
from typing import Iterable
from urllib.parse import quote

import requests
import wikipediaapi
from loguru import logger
from tqdm.asyncio import tqdm as tqdm_async
from utils.constants import WIKI_API


def get_random_titles(lang: str, n: int, user_agent: str | None = None) -> list[str]:
    """Вернёт n случайных заголовков статей (namespace=0)."""
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent or "PsiloQA/0.1 (+https://example.org; contact: youremail@example.org)"})
    titles: list[str] = []
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


def _norm_title(title: str) -> str:
    return (title or "").replace(" ", "_")


def _full_url(lang: str, title: str) -> str:
    return f"https://{lang}.wikipedia.org/wiki/{quote(_norm_title(title), safe='_()')}"


async def _fetch_page_text_wikipediaapi(
    wiki: wikipediaapi.Wikipedia,
    lang: str,
    title: str,
    *,
    min_len: int,
    sem: asyncio.Semaphore,
    per_request_sleep: float = 0.0,
) -> dict | None:
    async with sem:
        if per_request_sleep:
            await asyncio.sleep(per_request_sleep)

        def _blocking_fetch():
            page = wiki.page(_norm_title(title))
            if not page.exists():
                return None
            text = (page.text or "").strip()
            if len(text) < min_len:
                return None
            out_title = page.title or title
            return {
                "language": lang,
                "title": out_title,
                "url": page.fullurl or _full_url(lang, out_title),
                "passage": text,
            }

        try:
            return await asyncio.to_thread(_blocking_fetch)
        except Exception:
            return None


async def get_random_pages_async(
    *,
    lang: str,
    titles: Iterable[str] | None = None,
    n: int = 0,
    min_len: int = 0,
    user_agent: str | None = None,
    show_progress: bool = True,
    per_request_sleep: float = 0.0,
    max_concurrency: int = 16,
) -> list[dict]:
    if titles is None:
        title_list = get_random_titles(lang, n, user_agent=user_agent)
    else:
        title_list = list(titles)

    ua = user_agent or "PsiloQA/0.1 (+https://example.org; contact: youremail@example.org)"
    wiki = wikipediaapi.Wikipedia(user_agent=ua, language=lang)

    sem = asyncio.Semaphore(max_concurrency)
    coros = [
        _fetch_page_text_wikipediaapi(
            wiki,
            lang,
            t,
            min_len=min_len,
            sem=sem,
            per_request_sleep=per_request_sleep,
        )
        for t in title_list
    ]

    results: list[dict] = []
    if show_progress:
        for done in tqdm_async.as_completed(coros, total=len(coros), desc=f"{lang}: fetching full pages", leave=False):
            row = await done
            if row:
                results.append(row)
    else:
        fetched = await asyncio.gather(*coros, return_exceptions=False)
        for row in fetched:
            if row:
                results.append(row)

    logger.success(f"[{lang}] kept {len(results)}/{len(title_list)} full pages (min_len={min_len})")
    return results
