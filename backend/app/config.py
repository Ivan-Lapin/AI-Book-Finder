from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Book Finder"
    qdrant_url: str = Field(default="http://qdrant:6333")
    qdrant_collection: str = Field(default="books")
    embedding_model_name: str = Field(default="intfloat/multilingual-e5-base")
    default_limit: int = Field(default=5)
    upload_dir: str = Field(default="/shared/uploads")
    default_catalog_path: str = Field(default="/shared/data/books.json")
    
    qdrant_startup_timeout_sec: int = 60
    qdrant_startup_poll_interval_sec: float = 2.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
