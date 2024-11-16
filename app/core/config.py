# app/core/config.py
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    BASE_URL: str = "https://api.production.cloudios.flowfact-prod.cloud"
    ELASTICSEARCH_HOST: str = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
    API_KEY: str = os.getenv("API_KEY", "your_default_api_key")
    TIMEOUT: int = int(os.getenv("TIMEOUT", "30"))

settings = Settings()
