from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class HFSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="HF_",
        extra="ignore",
    )

    token: SecretStr


class OpenAISettings(BaseSettings):
    openai_api_key: SecretStr
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.0)
    base_url: str = Field(default="https://api.openai.com/v1")
    max_retries: int = 3
    max_concurrency: int = 8

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore",
    )


class QAGeneratorOpenAISettings(OpenAISettings):
    system_prompt_path: Path = Field(default=Path("psilo/dataset/prompts/qa_generator_system_prompt.txt"))

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="QA_GENERATOR_",
        extra="ignore",
    )


class AnnotatorOpenAISettings(OpenAISettings):
    system_prompt_path: Path = Field(default=Path("psilo/dataset/prompts/annotator_system_prompt.txt"))
    template_path: Path = Field(default=Path("psilo/dataset/prompts/annotator_template.txt"))

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ANNOTATOR_",
        extra="ignore",
    )


class FilterOpenAISettings(OpenAISettings):
    system_prompt_subjective_incomplete_path: Path = Field(default=Path("psilo/dataset/prompts/filter_subjective_incomplete.txt"))
    system_prompt_refusals_path: Path = Field(default=Path("psilo/dataset/prompts/filter_refusals.txt"))

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FILTER_",
        extra="ignore",
    )
