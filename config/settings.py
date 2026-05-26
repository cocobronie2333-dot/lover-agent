from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # Chroma 配置
    chroma_db_path: str = "./data/chroma"
    chroma_collection_name: str = "chat_records"

    # 嵌入模型配置
    embedding_model: str = "bge-small-zh"
    embedding_model_path: str = "./models/bge-small-zh"

    # LLM 配置
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = "sk-placeholder"
    llm_model: str = "gpt-4o-mini"

    # 日志
    log_level: str = "INFO"

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()