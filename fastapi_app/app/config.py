import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"



class Settings(BaseSettings):
    MONGO_URI: str
    DATABASE_NAME: str = "air_quality_db"
    MODEL_DIR: str
    
    DEFAULT_FEATURES: list = ['PM2.5', 'TSP', 'O3', 'CO', 'NO2', 'SO2', 'Temperature', 'Humidity']
    STATION_2_FEATURES: list = ['PM2.5', 'TSP', 'O3', 'NO2', 'SO2', 'Temperature', 'Humidity']

    model_config = SettingsConfigDict(
        env_file=ENV_PATH, 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

settings = Settings()