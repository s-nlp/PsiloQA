from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseSettings):
    openai_api_key: SecretStr
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=1.0)
    base_url: str = Field(default="https://api.openai.com/v1")
    max_retries: int = 3
    max_concurrency: int = 8


class QAGeneratorOpenAISettings(OpenAISettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="QA_GENERATOR_",
        extra="ignore",
    )
    system_prompt_path: Path = Field(default=Path("psilo/prompts/qa_generator_system_prompt.txt"))


class AnnotatorOpenAISettings(OpenAISettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ANNOTATOR_",
        extra="ignore",
    )
    system_prompt_path: Path = Field(default=Path("psilo/prompts/annotator_system_prompt.txt"))
    template_path: Path = Field(default=Path("psilo/prompts/annotator_template.txt"))
