from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    LANGSMITH_API_KEY = str | None
    GOOGLE_API_KEY = str | None
    LANGSMITH_TRACING = bool | None = None


settings = Settings()
