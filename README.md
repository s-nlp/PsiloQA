# PsiloQA: generate multilingual span-level inconsistency detection data synthetically!
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

## Dataset
Parse random pages from Wikipedia:
```bash
uv run psilo dataset get_contexts --num-pages 10 --language en --language ru
```

Generate QA pairs:
```bash
OPENAI_API_KEY=... uv run psilo dataset generate_qa
```

Generate LLM hypotheses:
```bash
uv run psilo dataset generate_llm_answers
```

Annotate hypotheses
```bash
uv run psilo dataset annotate
```

Filter data
```bash
TBD
```

## Model
