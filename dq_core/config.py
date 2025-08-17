"""Configuration management with Pydantic models and environment overrides."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class Environment(str, Enum):
    """Deployment environment."""
    
    LOCAL = "local"
    DEV = "dev"
    TEST = "test"
    STAGING = "staging"
    PROD = "prod"


class StorageType(str, Enum):
    """Storage backend types."""
    
    SPARK = "spark"
    SNOWFLAKE = "snowflake"
    S3 = "s3"
    LOCAL = "local"


class DatabaseConfig(BaseModel):
    """Database configuration."""
    
    host: str = "localhost"
    port: int = 5432
    database: str = "dqmetrics"
    username: str = "dquser"
    password: str = "dqpass"
    pool_size: int = 10
    echo: bool = False


class SnowflakeConfig(BaseModel):
    """Snowflake connection configuration."""
    
    account: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    warehouse: Optional[str] = None
    database: Optional[str] = None
    schema: Optional[str] = "PUBLIC"
    role: Optional[str] = None


class S3Config(BaseModel):
    """S3/MinIO configuration."""
    
    endpoint_url: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    bucket: str = "dq-data"
    region: str = "us-east-1"
    use_ssl: bool = True


class SparkConfig(BaseModel):
    """Spark configuration."""
    
    app_name: str = "DataSentinel"
    master: str = "local[*]"
    executor_memory: str = "2g"
    driver_memory: str = "2g"
    shuffle_partitions: int = 200
    adaptive_enabled: bool = True
    additional_configs: Dict[str, Any] = Field(default_factory=dict)


class AlertConfig(BaseModel):
    """Alert configuration."""
    
    enabled: bool = True
    slack_webhook: Optional[str] = None
    pagerduty_key: Optional[str] = None
    email_smtp_host: Optional[str] = None
    email_from: Optional[str] = None
    email_to: List[str] = Field(default_factory=list)


class DQScoreWeights(BaseModel):
    """DQ Score dimension weights."""
    
    accuracy: float = 0.4
    reliability: float = 0.3
    stewardship: float = 0.15
    usability: float = 0.15
    
    @field_validator("accuracy", "reliability", "stewardship", "usability")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Weight must be between 0 and 1")
        return v
    
    def validate_sum(self) -> None:
        total = self.accuracy + self.reliability + self.stewardship + self.usability
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


class DatasetConfig(BaseModel):
    """Dataset configuration."""
    
    name: str
    storage_type: StorageType
    location: str
    owner: Optional[str] = None
    sla_hours: Optional[float] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    checks: List[str] = Field(default_factory=list)
    profiling_enabled: bool = True
    anomaly_detection_enabled: bool = True
    sampling_rate: float = 1.0
    partition_column: Optional[str] = None
    
    @field_validator("sampling_rate")
    @classmethod
    def validate_sampling_rate(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError("Sampling rate must be between 0 and 1")
        return v


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DQ_",
        case_sensitive=False,
    )
    
    environment: Environment = Environment.LOCAL
    debug: bool = False
    log_level: str = "INFO"
    
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    snowflake: SnowflakeConfig = Field(default_factory=SnowflakeConfig)
    s3: S3Config = Field(default_factory=S3Config)
    spark: SparkConfig = Field(default_factory=SparkConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    score_weights: DQScoreWeights = Field(default_factory=DQScoreWeights)
    
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    dashboard_port: int = 8501
    
    result_path: Path = Path("./dq_results")
    history_retention_days: int = 90
    
    pii_redaction_enabled: bool = True
    audit_logging_enabled: bool = True
    
    prometheus_enabled: bool = False
    prometheus_port: int = 9090
    
    openlineage_enabled: bool = False
    openlineage_endpoint: Optional[str] = None


class Config(BaseModel):
    """Full configuration including settings and datasets."""
    
    settings: Settings = Field(default_factory=Settings)
    datasets: List[DatasetConfig] = Field(default_factory=list)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        
        # Merge with environment settings
        settings = Settings()
        if "settings" in data:
            settings = Settings(**data["settings"])
        
        return cls(
            settings=settings,
            datasets=[DatasetConfig(**d) for d in data.get("datasets", [])]
        )
    
    def get_dataset(self, name: str) -> Optional[DatasetConfig]:
        """Get dataset configuration by name."""
        for dataset in self.datasets:
            if dataset.name == name:
                return dataset
        return None


def load_config(path: Optional[Path] = None) -> Config:
    """Load configuration from file or environment."""
    if path and path.exists():
        return Config.from_yaml(path)
    return Config()


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()