import ast
import asyncio
import json
import random

from dataset.settings import QAGeneratorOpenAISettings
from loguru import logger
from tqdm import tqdm
from utils.constants import LONG_ANSWER_CONSTRAINT, SHORT_ANSWER_CONSTRAINT
from utils.io import read_text


def build_content(prompt_template: str, passage: str) -> str:
    return prompt_template.format(
        answer_length_constraint=SHORT_ANSWER_CONSTRAINT if random.random() < 0.66 else LONG_ANSWER_CONSTRAINT,
        p=passage,
    )


def parse_model_output(raw: str) -> list[dict]:
    raw = raw.replace("```python", "").replace("```", "")
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except Exception:
        obj = ast.literal_eval(raw)
    if not isinstance(obj, list):
        raise ValueError("Model output is not a list")
    norm: list[dict] = []
    for x in obj:
        if not isinstance(x, dict):
            raise ValueError("List element is not an object")
        q = (x.get("question") or "").strip()
        a = (x.get("answer") or "").strip()
        c = (x.get("complexity") or "").strip()
        if not q or not a:
            continue
        norm.append({"question": q, "answer": a, "complexity": c})
    return norm


async def call_openai_once_async(
    client,
    model: str,
    content: str,
    temperature: float,
) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


async def generate_qa_for_contexts(client, rows: list[dict], settings: QAGeneratorOpenAISettings) -> list[dict]:
    """
    Asynchronous version. Sends multiple requests in parallel.
    Keeps the same prompt/parse behavior as your notebook.

    - client: AsyncOpenAI()
    - rows: dicts with 'summary' (or 'passage'), 'title', 'language', 'url'
    - keep_order: if True, results preserve input order
    """
    sem = asyncio.Semaphore(settings.max_concurrency)
    pbar = tqdm(total=len(rows), desc="QA generation...", leave=False)
    system_prompt = read_text(settings.system_prompt_path)

    async def process_one(idx: int, r: dict) -> list[dict]:
        passage = r.get("passage")
        content = build_content(system_prompt, passage)

        for attempt in range(1, settings.max_retries + 1):
            try:
                async with sem:
                    raw = await call_openai_once_async(
                        client=client,
                        model=settings.model,
                        content=content,
                        temperature=settings.temperature,
                    )
                triples = parse_model_output(raw)
                out_rows: list[dict] = []
                for t in triples:
                    out_rows.append(r | {"question": t["question"], "gold_answer": t["answer"], "complexity": t.get("complexity", ""), "model": settings.model})
                return out_rows
            except Exception as e:
                # exact retry behavior is up to you; keeping it minimal
                logger.warning(f"[retry {attempt}/{settings.max_retries}] {e}")
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
        logger.error(f"Failed for: {r.get('source_url')}")
        return []
        # pbar is updated in finally below

    async def wrapped(idx: int, r: dict):
        try:
            return idx, await process_one(idx, r)
        finally:
            if pbar:
                pbar.update(1)

    tasks = [asyncio.create_task(wrapped(i, r)) for i, r in enumerate(rows)]
    results = await asyncio.gather(*tasks)

    if pbar:
        pbar.close()

    results.sort(key=lambda x: x[0])  # restore input order
    flat: list[dict] = []
    for _, batch in results:
        flat.extend(batch)
    return flat
