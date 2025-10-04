# PsiloQA: Generate multilingual span-level inconsistency detection data synthetically!
![PsiloQA logo](images/logo.png)

## Installation
Install uv:
```bash
pip install uv
```

Install dependencies:
```bash
uv sync --no-dev
```

Copy env.example and fill env variables:
```bash
cp env.example .env
```

## PsiloQA Dataset Generation Pipeline
The **PsiloQA pipeline** automates the construction of a **multilingual, span-level hallucination detection dataset with contexts** — from sampling Wikipedia passages to generating Q&A, producing model hypotheses, annotating hallucinated spans, and filtering the results.

It consists of five sequential stages:
1. **Contexts** — parse random Wikipedia pages as input passages for QA generation.  
2. **QA pairs** — generate questions and answers of varying complexity using an OpenAI model.  
3. **LLM hypotheses** — produce candidate model answers for evaluation.  
4. **Annotation** — mark hallucinated spans in model hypotheses using an OpenAI-based annotator.  
5. **Filtering** — automatically clean data via heuristic and LLM-based filters.

Each stage can be run individually, or you can execute the full pipeline with a single command:

```bash
uv run psilo dataset pipeline --num-pages 10 --language ru --language en --limit 100
```

All API keys and model settings are managed via the `.env` file (`QA_GENERATOR_`, `ANNOTATOR_`, and `FILTER_` prefixes).

### Contexts
The first step in PsiloQA pipeline is getting contexts for QA generation. You can use your own, or, as in out paper, parse random pages from Wikipedia as input contexts. Just run the following command with languages you need. If no `--language` list specified, it will parse random pages for 14 languages presented in our paper. `--num-pages` determines how many contexts to parse from Wikipedia.
```bash
uv run psilo dataset get_contexts --num-pages 10 --language ru --language en
```

### QA pairs
Next step is question and answer generation for the obtained contexts. The script generates three questions of different complexity based on provided contexts. Fill `QA_GENERATOR` settings in `.env` file to use this script. By default, `gpt-4o` is used. Feel free to use another models by providing another model name through `QA_GENERATOR` setting in `.env`.
```bash
uv run psilo dataset generate_qa
```

### LLM hypotheses
Generate LLM hypotheses:
```bash
uv run psilo dataset generate_hypotheses
```

All available models are listed in `psilo/dataset/answer_generator/models`. You can add any new Hugging Face model by implementing a runner class that inherits from either:
- `RunnerWithChatTemplate` — if the tokenizer supports chat templates, or
- `RunnerWithCustomTemplate` — if it does not.
Some models require a Hugging Face access token. Make sure to provide `HF_TOKEN` in your `.env` file — models that need it will be skipped if the token is missing.

### Hypotheses annotation
Annotate hypotheses (fill `ANNOTATOR_OPENAI_API_KEY` variable in .env):
```bash
uv run psilo dataset annotate_hypotheses
```

### Filtering
The annotation process includes two filtering stages. Heuristic-based filters ensure structural correctness — they verify that all opening tags have corresponding closing tags, that there are no nested tags, and perform other automated pre-checks. LLM-based filters remove samples with subjective or incomplete questions, as well as cases where the model refuses to answer. For LLM-based filter, fill `FILTER_OPENAI_API_KEY` variable in .env
```bash
uv run psilo dataset filter
```

