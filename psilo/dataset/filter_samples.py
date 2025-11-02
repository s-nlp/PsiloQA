import asyncio
import re

from dataset.settings import FilterOpenAISettings
from loguru import logger
from openai import AsyncOpenAI
from tqdm import tqdm
from utils.io import read_text
from utils.openai import call_openai_once_async


def compare_input_and_output(model_input: str, model_output: str) -> bool:
    cleaned_output = re.sub(r"\[HAL\]|\[/HAL\]", "", model_output or "")
    return cleaned_output.strip() == (model_input or "").strip()


def check_hal_tags_no_nesting(text: str) -> bool:
    index = 0
    length = len(text or "")
    while index < length:
        open_tag_pos = text.find("[HAL]", index)
        if open_tag_pos == -1:
            break
        close_tag_pos = text.find("[/HAL]", open_tag_pos + len("[HAL]"))
        if close_tag_pos == -1:
            return False
        nested_tag_pos = text.find("[HAL]", open_tag_pos + len("[HAL]"), close_tag_pos)
        if nested_tag_pos != -1:
            return False
        index = close_tag_pos + len("[/HAL]")
    remaining_text = text[index:]
    if "[/HAL]" in remaining_text:
        return False
    return True


def check_empty_tag(model_output: str) -> bool:
    texts_inside_tags = re.findall(r"\[HAL\](.*?)\[/HAL\]", model_output or "")
    for text in texts_inside_tags:
        if len(text.strip()) == 0:
            return False
    return True


def filter_rows_with_heuristics(rows: list[dict]) -> list[dict]:
    filtered = []
    for row in rows:
        model_input = row.get("hypothesis")
        model_output = row.get("annotation")

        if not model_input or not model_output:
            continue

        if not check_hal_tags_no_nesting(model_output):
            logger.warning(f"[HAL] tags no nesting: {model_output}")
            continue
        if not check_empty_tag(model_output):
            logger.warning(f"Empty [HAL][/HAL]: {model_output}")
            continue
        if not compare_input_and_output(model_input, model_output):
            logger.warning(f"Annotated output is inconsistent with the initial output: {model_output}")
            continue

        filtered.append(row)
    logger.info(f"After heuristics: {len(filtered)}/{len(rows)} kept")
    return filtered


def parse_subjective_incomplete(raw: str) -> tuple[str, bool]:
    text = (raw or "").strip().upper()
    text = re.sub(r"[^A-Z_]", "", text)
    print(text)

    if "INCOMPLETE" in text:
        return False
    if "SUBJECTIVE" in text:
        return False
    if "NORMAL" in text:
        return True
    return False


def parse_refusal_bool(raw: str) -> bool:
    text = (raw or "").strip().upper()
    if text.startswith("TRUE"):
        return False
    if text.startswith("FALSE"):
        return True
    return False


async def apply_filter_prompt_async(
    client: AsyncOpenAI,
    rows: list[dict],
    system_prompt: str,
    settings: "FilterOpenAISettings",
    desc: str,
    parse_fn,
    user_field: str,
) -> list[dict]:
    sem = asyncio.Semaphore(settings.max_concurrency)
    pbar = tqdm(total=len(rows), desc=desc, leave=False)

    async def process_one(r: dict) -> tuple[int, bool]:
        user_text = r.get(user_field) or ""
        for attempt in range(1, settings.max_retries + 1):
            try:
                async with sem:
                    print(user_text)
                    raw = await call_openai_once_async(
                        client=client,
                        model=settings.model,
                        user_prompt=user_text,
                        temperature=settings.temperature,
                        system_prompt=system_prompt,
                    )

                keep = parse_fn(raw)
                return r.get("id", -1), keep
            except Exception as e:
                logger.warning(f"[{desc}] retry {attempt}/{settings.max_retries}: {e}")
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
        logger.error(f"[{desc}] failed after retries for id={r.get('id')}")
        return r.get("id", -1), False

    tasks = [asyncio.create_task(process_one(r)) for r in rows]
    results = await asyncio.gather(*tasks)
    pbar.close()

    kept_ids = {rid for rid, keep in results if keep}
    return [r for r in rows if r.get("id", -2) in kept_ids or (-1 in kept_ids and r.get("id") is None)]


async def run_llm_filters_async(
    client: AsyncOpenAI,
    rows: list[dict],
    settings: "FilterOpenAISettings",
) -> list[dict]:
    prompt_incomplete = read_text(settings.system_prompt_subjective_incomplete_path)
    prompt_refusals = read_text(settings.system_prompt_refusals_path)

    after_questions = await apply_filter_prompt_async(
        client=client,
        rows=rows,
        system_prompt=prompt_incomplete,
        settings=settings,
        desc="Subjective / Incomplete",
        parse_fn=parse_subjective_incomplete,
        user_field="question",
    )
    logger.info(f"After Subjective/Incomplete: {len(after_questions)}/{len(rows)} kept")

    after_answers = await apply_filter_prompt_async(
        client=client,
        rows=after_questions,
        system_prompt=prompt_refusals,
        settings=settings,
        desc="Refusals",
        parse_fn=parse_refusal_bool,
        user_field="hypothesis",
    )
    logger.info(f"After Refusals: {len(after_answers)}/{len(after_questions)} kept")

    return after_answers
