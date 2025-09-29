import asyncio

from dataset.settings import AnnotatorOpenAISettings
from loguru import logger
from tqdm.auto import tqdm
from utils.io import read_text


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


async def annotate_hypotheses(client, rows: list[dict], settings: AnnotatorOpenAISettings) -> list[dict]:
    system_prompt = read_text(settings.system_prompt_path)
    template = read_text(settings.template_path)

    sem = asyncio.Semaphore(settings.max_concurrency)
    pbar = tqdm(total=len(rows), desc="Hypotheses annotation...", leave=False)

    async def process_one(idx: int, r: dict) -> list[dict]:
        passage = r.get("passage")
        question = r.get("question")
        gold_answer = r.get("gold_answer")
        hypothesis = r.get("hypothesis")

        for attempt in range(1, settings.max_retries + 1):
            try:
                async with sem:
                    raw = await call_openai_once_async(
                        client=client,
                        model=settings.model,
                        system_prompt=system_prompt,
                        user_prompt=template.format(q=question, p=passage, ga=gold_answer, a=hypothesis),
                        temperature=settings.temperature,
                    )
                output = parse_model_output(raw)
                return r | {
                    "annotator_model": settings.model,
                    "annotated": output,
                }
            except Exception as e:
                logger.warning(f"[retry {attempt}/{settings.max_retries}] {e}")
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
        return []

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

    results.sort(key=lambda x: x[0])

    return [r for _, r in results]
