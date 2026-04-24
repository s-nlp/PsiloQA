# PsiloQA: Generate multilingual span-level inconsistency detection data synthetically!
[![Dataset on Hugging Face](https://img.shields.io/badge/Dataset-HuggingFace-blue.svg)](https://huggingface.co/datasets/s-nlp/PsiloQA)
![PsiloQA logo](images/logo.png)

## About

**PsiloQA** is the largest dataset for training and evaluating systems on **multilingual span-level hallucination detection with retrieved context**. 

It offers:

- 🧠 **An automated and scalable pipeline** for generating, annotating, and filtering data for hallucination detection tasks  
- 🌍 **A large multilingual dataset** covering **14 languages** with high-quality, fine-grained span-level hallucination annotations for multiple open-source LLMs  
- 📊 **Comprehensive empirical evaluations** of various state-of-the-art span-level hallucination detection methods across 14 languages  

You can explore or download the dataset on Hugging Face:  
👉 **[s-nlp/PsiloQA](https://huggingface.co/datasets/s-nlp/PsiloQA)**

This repository contains the **full PsiloQA generation pipeline** — from sampling multilingual Wikipedia contexts to question–answer generation, LLM hypothesis production, annotation, and filtering.

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
uv run psilo dataset pipeline --num-pages 10 --language ru --language en --limit 100 --model Qwen/Qwen2.5-3B-Instruct
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
All available models are listed in `psilo/dataset/answer_generator/models`. You can add any new Hugging Face model by implementing a runner class that inherits from either:
- `RunnerWithChatTemplate` — if the tokenizer supports chat templates, or
- `RunnerWithCustomTemplate` — if it does not.
Some models require a Hugging Face access token. Make sure to provide `HF_TOKEN` in your `.env` file — models that need it will be skipped if the token is missing.
```bash
uv run psilo dataset generate_hypotheses
```

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

### Uncertainty
Use the following command to run uncertainty quantification methods:
```bash
uv run psilo methods uncertainty
```

## Citation
```
@inproceedings{rykov-etal-2025-models,
    title = "When Models Lie, We Learn: Multilingual Span-Level Hallucination Detection with {P}silo{QA}",
    author = "Rykov, Elisei  and
      Petrushina, Kseniia  and
      Savkin, Maksim  and
      Olisov, Valerii  and
      Vazhentsev, Artem  and
      Titova, Kseniia  and
      Panchenko, Alexander  and
      Konovalov, Vasily  and
      Belikova, Julia",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.626/",
    doi = "10.18653/v1/2025.findings-emnlp.626",
    pages = "11663--11682",
    ISBN = "979-8-89176-335-7",
    abstract = "Hallucination detection remains a fundamental challenge for the safe and reliable deployment of large language models (LLMs), especially in applications requiring factual accuracy. Existing hallucination benchmarks often operate at the sequence level and are limited to English, lacking the fine-grained, multilingual supervision needed for comprehensive evaluation. In this work, we introduce PsiloQA, a large-scale, multilingual dataset annotated with span-level hallucinations across 14 languages. PsiloQA is constructed through an automated three-stage pipeline: generating question{--}answer pairs from Wikipedia using GPT-4o, eliciting potentially hallucinated answers from diverse LLMs in a no-context setting, and automatically annotating hallucinated spans using GPT-4o by comparing against golden answers and retrieved context. We evaluate a wide range of hallucination detection methods-including uncertainty quantification, LLM-based tagging, and fine-tuned encoder models-and show that encoder-based models achieve the strongest performance across languages. Furthermore, PsiloQA demonstrates effective cross-lingual generalization and supports robust knowledge transfer to other benchmarks, all while being significantly more cost-efficient than human-annotated datasets. Our dataset and results advance the development of scalable, fine-grained hallucination detection in multilingual settings."
}
```

