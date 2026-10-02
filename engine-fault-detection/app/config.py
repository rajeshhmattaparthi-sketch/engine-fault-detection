from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=("settings_",))

    database_url: str = "sqlite+aiosqlite:///./engine.db"
    model_path: str = "models/fault_model.joblib"
    train_samples: int = 6000
    random_seed: int = 42


settings = Settings()
