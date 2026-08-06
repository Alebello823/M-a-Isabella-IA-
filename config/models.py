# config/models.py
from typing import Optional
from pathlib import Path
import pydantic
BaseModel = pydantic.BaseModel
Field = pydantic.Field
field_validator = pydantic.field_validator

class AppConfig(BaseModel):
    name: str = "Mía Isabella"
    version: str = "0.1.0"
    environment: str = "development"
    default_language: str = "es"

class StorageConfig(BaseModel):
    database_path: Path = Path("data/mia_isabella.db")
    vector_store_path: Path = Path("data/vectors")
    graph_store_path: Optional[Path] = None

    @field_validator("database_path", "vector_store_path", mode="before")
    def resolve_path(cls, v):
        return Path(v).expanduser().resolve()

class InferenceConfig(BaseModel):
    default_model: str = "phi3:mini"
    default_model_path: Path = Path("models/phi3-mini-4k-instruct.Q4_K_M.gguf")
    max_tokens: int = 512
    temperature: float = 0.7
    fallback_model: Optional[str] = "tinyllama"

    @field_validator("default_model_path", mode="before")
    def resolve_path(cls, v):
        return Path(v).expanduser().resolve()

class TelemetryConfig(BaseModel):
    log_level: str = "INFO"
    enable_performance_tracing: bool = True
    metrics_retention_days: int = 30

class IdentityConfig(BaseModel):
    default_ai_name: str = "Mía Isabella"
    default_relationship: str = "Pa"

class HardwareConfig(BaseModel):
    auto_detect: bool = True
    manual_profile: Optional[str] = None

class SyncConfig(BaseModel):
    enabled: bool = False
    conflict_strategy: str = "manual"

class Settings(BaseModel):
    app: AppConfig = AppConfig()
    storage: StorageConfig = StorageConfig()
    inference: InferenceConfig = InferenceConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    identity: IdentityConfig = IdentityConfig()
    hardware: HardwareConfig = HardwareConfig()
    sync: SyncConfig = SyncConfig()
