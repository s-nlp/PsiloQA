# psilo/dataset/cli.py
import asyncio
import json
from pathlib import Path

import typer
from dataset.answer_generator.batching import assign_runners_by_language
from dataset.answer_generator.registry import all_runners, sample_runner_for_language
from dataset.generate_qa import generate_qa_for_summaries
from dataset.wiki_contexts import get_random_pages
from loguru import logger
from openai import AsyncOpenAI
from tqdm import tqdm
from utils.io import read_jsonl, read_text, write_jsonl
from dataset.answer_generator import models  # noqa: F401

app = typer.Typer(help="PsiloQA Generation Pipeline")


@app.command("get_contexts")
def get_contexts(
    languages: list[str] = typer.Option(
        ["en"], "--language", "-l", help="ISO codes, e.g. en ru de"
    ),
    num_pages: int = typer.Option(100, "--num-pages", "-n", help="Pages per language"),
    min_string_length: int = typer.Option(
        100, "--min-len", help="Min length of page text"
    ),
    seed: int = typer.Option(42, "--seed"),
    output_path: Path = typer.Option(
        "data/raw/output.jsonl", "--out", help="Path to store the contexts"
    ),
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output_path.open("w", encoding="utf-8") as f:
        for lang in languages:
            logger.info(f"Sampling {num_pages} pages from {lang} Wikipedia...")
            rows = get_random_pages(lang=lang, n=num_pages, min_len=min_string_length)
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
        help="Path to Wikipedia contexts JSONL (title, summary, language, url)",
    ),
    output_path: Path = typer.Option(
        Path("data/qa/output.jsonl"), "--out", help="Path to write QA JSONL"
    ),
    prompt_file: Path = typer.Option(
        Path("psilo/prompts/wiki_qa.txt"),
        "--prompt-file",
        help="Path to the EXACT prompt template used in the notebook",
    ),
    openai_api_key: str | None = typer.Option(
        None, "--openai-api-key", envvar="OPENAI_API_KEY"
    ),
    model: str = typer.Option(
        "gpt-4o-mini", "--model", help="OpenAI model id (e.g., o3-mini, gpt-4o-mini)"
    ),
    temperature: float = typer.Option(1.0, "--temperature"),
    seed: int | None = typer.Option(None, "--seed"),
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Reading: {input_path}")
    rows = read_jsonl(str(input_path))
    logger.info(f"Rows: {len(rows)}")

    prompt_template = read_text(str(prompt_file))

    async def _run():
        client = AsyncOpenAI(api_key=openai_api_key)
        return await generate_qa_for_summaries(
            client=client,
            model=model,
            rows=rows,
            prompt_template=prompt_template,
            temperature=temperature,
            seed=seed,
            show_progress=True,
            max_retries=3,
            max_concurrency=8,
            keep_order=True,
        )

    qa_rows = asyncio.run(_run())
    logger.info(f"Writing QA: {len(qa_rows)} → {output_path}")
    write_jsonl(str(output_path), qa_rows, "a")
    logger.success("Done.")


@app.command("generate_llm_answers")
def generate_llm_answers(
    input_path: Path = typer.Option(
        "data/qa/output.jsonl",
        "--in",
        help="Path to QA JSONL with at least {question, language}",
    ),
    output_path: Path = typer.Option("data/interim/hypotheses.jsonl", "--out"),
    limit: int | None = typer.Option(None, "--limit", help="Process only N samples"),
    seed: int | None = typer.Option(42, "--seed"),
):
    rows = read_jsonl(str(input_path))
    if limit:
        rows = rows[:limit]
    logger.info(f"Loaded {len(rows)} QA rows")

    buckets = assign_runners_by_language(
        samples=rows,
        choose_runner_for_lang=sample_runner_for_language,
        seed=seed,
    )
    if not buckets:
        logger.error("No runners matched the sample languages.")
        raise typer.Exit(1)

    logger.info("Runners selected per language:")
    for rid, batch in buckets.items():
        langs = {(s.get("language") or "").lower() for _, s in batch}
        logger.info(f"  {rid}: {len(batch)} samples, langs={sorted(langs)}")

    outputs: list[dict] = []
    # Iterate per runner and generate
    for rid, batch in buckets.items():
        runner = all_runners()[rid]
        logger.info(f"Loading runner: {rid}")
        runner.load()

        questions = [s["question"] for _, s in batch]
        pbar = tqdm(total=len(batch), desc=f"{rid}", leave=False)

        for (idx, sample), res in zip(batch, runner.answer_batch(questions)):
            out = {
                "type": "hypothesis",
                "language": sample.get("language"),
                "question": sample["question"],
                "gold_answer": sample.get("answer"),
                "model_id": rid,
                "hypothesis": res.text,
                "gen_meta": res.meta,
            }
            outputs.append(out)
            pbar.update(1)
        pbar.close()

    logger.info(f"Writing {len(outputs)} rows → {output_path}")
    write_jsonl(str(output_path), outputs)
    logger.success("Done.")

    print(buckets.keys())


@app.command("annotate")
def cmd_annotate(): ...


@app.command("filter")
def cmd_filter(): ...
