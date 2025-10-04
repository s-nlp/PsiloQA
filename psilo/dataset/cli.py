# psilo/dataset/cli.py
import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from dataset.annotator import annotate_hypotheses
from dataset.filter_samples import filter_rows_with_heuristics, run_llm_filters_async
from dataset.generate_qa import generate_qa_for_contexts
from dataset.settings import AnnotatorOpenAISettings, FilterOpenAISettings, QAGeneratorOpenAISettings
from dataset.wiki_contexts import get_random_pages_async
from loguru import logger
from openai import AsyncOpenAI
from tqdm import tqdm
from utils.constants import DEFAULT_LANGUAGES
from utils.io import read_jsonl, write_jsonl

app = typer.Typer(help="PsiloQA Generation Pipeline")


@app.command("get_contexts")
def get_contexts(
    languages: Annotated[list[str], typer.Option("--language", "-l", help="ISO codes, e.g. en ru de")] = DEFAULT_LANGUAGES,
    num_pages: Annotated[int, typer.Option("--num-pages", "-n", help="Pages per language")] = 100,
    min_string_length: Annotated[int, typer.Option("--min-len", help="Min length of page text")] = 100,
    output_path: Annotated[Path, typer.Option("--out", help="Path to store the contexts")] = Path("data/raw/output.jsonl"),
    max_concurrency: Annotated[int, typer.Option(help="Concurrent requests per language")] = 16,
    per_request_sleep: Annotated[float, typer.Option("--sleep", help="Optional delay (seconds) before each request")] = 0.0,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output_path.open("w", encoding="utf-8") as f:
        for lang in languages:
            logger.info(f"Sampling {num_pages} pages from {lang} Wikipedia (async)...")

            async def _run_one_lang():
                return await get_random_pages_async(
                    lang=lang,
                    n=num_pages,
                    min_len=min_string_length,
                    show_progress=True,
                    per_request_sleep=per_request_sleep,
                    max_concurrency=int(max_concurrency)
                )

            rows = asyncio.run(_run_one_lang())

            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            total += len(rows)
            logger.success(f"{lang}: saved {len(rows)} rows")
    logger.success(f"Saved total {total} rows → {output_path}")



@app.command("generate_qa")
def generate_qa(
    input_path: Path = typer.Option(
        Path("data/raw/output.jsonl"),
        "--in",
        help="Path to Wikipedia contexts JSONL",
    ),
    output_path: Path = typer.Option(Path("data/qa/output.jsonl"), "--out", help="Path to write QA JSONL"),
):
    logger.info(f"Reading: {input_path}")
    rows = read_jsonl(str(input_path))
    logger.info(f"Rows: {len(rows)}")

    settings: QAGeneratorOpenAISettings = QAGeneratorOpenAISettings()

    async def _run():
        client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value(), base_url=settings.base_url)
        return await generate_qa_for_contexts(client=client, rows=rows, settings=settings)

    qa_rows = asyncio.run(_run())

    logger.info(f"Writing QA: {len(qa_rows)} → {output_path}")
    write_jsonl(output_path, qa_rows)
    logger.success("Done.")


@app.command("generate_hypotheses")
def generate_hypotheses(
    input_path: Path = typer.Option(
        "data/qa/output.jsonl",
        "--in",
        help="Path to QA JSONL with at least {question, language}",
    ),
    output_path: Path = typer.Option("data/hypotheses/out.jsonl", "--out"),
    limit: int | None = typer.Option(None, "--limit", help="Process only N samples"),
):
    from dataset.answer_generator import models  # noqa: F401
    from dataset.answer_generator.batching import assign_runners_by_language
    from dataset.answer_generator.registry import all_runners, sample_runner_for_language
    from dataset.answer_generator.runner import RunnerType
    from huggingface_hub.utils import HfHubHTTPError, GatedRepoError, RepositoryNotFoundError

    rows = read_jsonl(str(input_path))
    if limit:
        rows = rows[:limit]
    logger.info(f"Loaded {len(rows)} QA rows")

    buckets = assign_runners_by_language(
        samples=rows,
        choose_runner_for_lang=sample_runner_for_language,
    )
    if not buckets:
        logger.error("No runners matched the sample languages.")
        raise typer.Exit(1)

    logger.info("Runners selected per language:")
    for rid, batch in buckets.items():
        langs = {(s.get("language") or "").lower() for _, s in batch}
        logger.info(f"  {rid}: {len(batch)} samples, langs={sorted(langs)}")

    for rid, batch in buckets.items():
        outputs: list[dict] = []

        runner = all_runners()[rid]
        logger.info(f"Loading runner: {rid}")
        try:
            runner.load()
        except (HfHubHTTPError, GatedRepoError, RepositoryNotFoundError) as E:
            logger.warning(f"Model {rid} not found or HF_TOKEN with the access is not provided")
            continue

        if runner.TYPE == RunnerType.VLLM:
            questions = [s["question"] for _, s in batch]
            pbar = tqdm(total=len(questions), desc=f"{rid}", leave=False)
            results = runner.answer_batch(questions)
            for (_, sample), res in zip(batch, results):
                outputs.append(sample | {"model_id": rid, "hypothesis": res.text, "gen_meta": res.meta})
                pbar.update(1)
            pbar.close()
        else:
            pbar = tqdm(total=len(batch), desc=f"{rid}", leave=False)
            for _, sample in batch:
                res = runner.answer_one(sample["question"])
                outputs.append(sample | {"model_id": rid, "hypothesis": res.text, "gen_meta": res.meta})
                pbar.update(1)
            pbar.close()

        logger.info(f"Writing {len(outputs)} rows → {output_path}")
        write_jsonl(output_path, outputs, mode="a")

        runner.destroy()

    logger.success("Done.")


@app.command("annotate_hypotheses")
def annotate(
    input_path: Path = typer.Option(
        Path("data/hypotheses/output.jsonl"),
        "--in",
        help="Path to Wikipedia contexts JSONL (title, summary, language, url)",
    ),
    output_path: Path = typer.Option(Path("data/annotated/output.jsonl"), "--out", help="Path to write annotated hypotheses"),
):
    logger.info(f"Reading: {input_path}")
    rows = read_jsonl(str(input_path))
    logger.info(f"Rows: {len(rows)}")

    settings = AnnotatorOpenAISettings()

    async def _run():
        client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        return await annotate_hypotheses(client=client, rows=rows, settings=settings)

    qa_rows = asyncio.run(_run())
    logger.info(f"Writing annotated hypotheses: {len(qa_rows)} → {output_path}")
    write_jsonl(output_path, qa_rows, "a")
    logger.success("Done.")


@app.command("filter")
def filter(
    input_path: Path = typer.Option(
        Path("data/annotated/output.jsonl"),
        "--in",
        help="Path to annotated hypotheses",
    ),
    output_path: Path = typer.Option(Path("data/filtered/output.jsonl"), "--out", help="Path to write filtered samples"),
):
    logger.info(f"Reading: {input_path}")
    rows = read_jsonl(str(input_path))

    logger.info("Stage 1: heuristic-based filtration")

    filtered_rows = filter_rows_with_heuristics(rows)

    logger.info("Stage 2: LLM-based filtration")

    settings = FilterOpenAISettings()

    async def _run():
        client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        return await run_llm_filters_async(client=client, rows=filtered_rows, settings=settings)

    filtered_rows = asyncio.run(_run())

    logger.info(f"Writing filtered samples: {len(filtered_rows)} → {output_path}")
    write_jsonl(output_path, filtered_rows, "a")
    logger.success("Done.")


@app.command("pipeline")
def pipeline(
    num_pages: int = typer.Option(10, "--num-pages", "-n", help="Pages per language for context sampling"),
    languages: list[str] = typer.Option(None, "--language", "-l", help="Languages (default: 14 from paper)"),
    limit: int | None = typer.Option(None, "--limit", help="Limit for QA/hypothesis generation"),
):
    """
    Run the full PsiloQA dataset generation pipeline step by step.
    """
    typer.echo("🚀 [1/5] Sampling Wikipedia contexts...")
    get_contexts(
        languages=languages or DEFAULT_LANGUAGES,
        num_pages=num_pages,
        min_string_length=100,
        output_path=Path("data/raw/output.jsonl"),
    )

    typer.echo("🧠 [2/5] Generating QA pairs...")
    generate_qa(
        input_path=Path("data/raw/output.jsonl"),
        output_path=Path("data/qa/output.jsonl"),
    )

    typer.echo("🤖 [3/5] Generating LLM hypotheses...")
    generate_hypotheses(
        input_path=Path("data/qa/output.jsonl"),
        output_path=Path("data/hypotheses/output.jsonl"),
        limit=limit,
    )

    typer.echo("🧩 [4/5] Annotating hypotheses...")
    annotate(
        input_path=Path("data/hypotheses/output.jsonl"),
        output_path=Path("data/annotated/output.jsonl"),
    )

    typer.echo("🧹 [5/5] Filtering annotated samples...")
    filter(
        input_path=Path("data/annotated/output.jsonl"),
        output_path=Path("data/filtered/output.jsonl"),
    )

    typer.echo("✅ PsiloQA pipeline completed successfully!")
