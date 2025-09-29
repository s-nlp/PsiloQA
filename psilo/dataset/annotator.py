import asyncio
import random

from loguru import logger
from tqdm.auto import tqdm


async def call_openai_once_async(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def parse_model_output(raw: str) -> list[dict]:
    raw = raw.replace("```python", "").replace("```", "")
    raw = raw.strip()
    raw = raw.replace("**", "")
    raw = raw.replace("Highlighted LLM Response:", "")
    return raw


async def annotate_hypotheses(
    client,
    model: str,
    rows: list[dict],
    system_prompt: str,
    prompt_template: str,
    temperature: float,
    seed: int | None = None,
    show_progress: bool = True,
    max_retries: int = 3,
    max_concurrency: int = 8,
    keep_order: bool = True,
) -> list[dict]:
    if seed is not None:
        random.seed(seed)

    sem = asyncio.Semaphore(max_concurrency)
    pbar = tqdm(total=len(rows), desc="Hypotheses annotation...", leave=False) if show_progress else None

    async def process_one(idx: int, r: dict) -> list[dict]:
        passage = r.get("passage")
        question = r.get("question")
        gold_answer = r.get("gold_answer")
        hypothesis = r.get("hypothesis")

        for attempt in range(1, max_retries + 1):
            try:
                async with sem:
                    raw = await call_openai_once_async(
                        client=client,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=prompt_template.format(q=question, p=passage, ga=gold_answer, a=hypothesis),
                        temperature=temperature,
                    )
                output = parse_model_output(raw)
                return {
                    "language": r.get("language"),
                    "title": r.get("title"),
                    "source_url": r.get("source_url"),
                    "passage": passage,
                    "question": question,
                    "gold_answer": gold_answer,
                    "hypothesis": hypothesis,
                    "complexity": r.get("complexity", ""),
                    "model": model,
                    "annotated": output,
                }
            except Exception as e:
                logger.warning(f"[retry {attempt}/{max_retries}] {e}")
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
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

    if keep_order:
        results.sort(key=lambda x: x[0])

    return [r for _, r in results]
