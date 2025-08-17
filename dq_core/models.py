"""Core data models for DataSentinel."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class CheckStatus(str, Enum):
    """Check execution status."""
    
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class CheckSeverity(str, Enum):
    """Check severity levels."""
    
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class DQDimension(str, Enum):
    """Data quality dimensions."""
    
    ACCURACY = "accuracy"
    RELIABILITY = "reliability"
    STEWARDSHIP = "stewardship"
    USABILITY = "usability"


class Dataset(BaseModel):
    """Dataset metadata."""
    
    id: UUID = Field(default_factory=uuid4)
    name: str
    storage_type: str
    location: str
    owner: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class Check(BaseModel):
    """Data quality check definition."""
    
    id: UUID = Field(default_factory=uuid4)
    name: str
    type: str
    description: Optional[str] = None
    severity: CheckSeverity = CheckSeverity.MEDIUM
    dimension: DQDimension = DQDimension.ACCURACY
    parameters: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    
    class Config:
        use_enum_values = True


class CheckResult(BaseModel):
    """Individual check execution result."""
    
    check_id: UUID
    check_name: str
    status: CheckStatus
    severity: CheckSeverity
    dimension: DQDimension
    passed_records: Optional[int] = None
    failed_records: Optional[int] = None
    total_records: Optional[int] = None
    pass_rate: Optional[float] = None
    error_message: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: Optional[int] = None
    
    @field_validator("pass_rate")
    @classmethod
    def validate_pass_rate(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not 0 <= v <= 1:
            raise ValueError("Pass rate must be between 0 and 1")
        return v
    
    class Config:
        use_enum_values = True


class ProfileMetrics(BaseModel):
    """Column profiling metrics."""
    
    column_name: str
    data_type: str
    count: int
    null_count: int
    null_percentage: float
    distinct_count: int
    distinct_percentage: float
    min_value: Optional[Union[float, str]] = None
    max_value: Optional[Union[float, str]] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std_dev: Optional[float] = None
    quantiles: Dict[str, float] = Field(default_factory=dict)
    top_values: List[Dict[str, Any]] = Field(default_factory=list)
    histogram: Optional[Dict[str, Any]] = None


class AnomalyResult(BaseModel):
    """Anomaly detection result."""
    
    metric_name: str
    current_value: float
    expected_range: tuple[float, float]
    z_score: Optional[float] = None
    is_anomaly: bool
    confidence: float
    method: str = "statistical"
    details: Dict[str, Any] = Field(default_factory=dict)


class DimensionScore(BaseModel):
    """Score for a single DQ dimension."""
    
    dimension: DQDimension
    score: float
    weight: float
    weighted_score: float
    passed_checks: int
    failed_checks: int
    total_checks: int
    signals: List[str] = Field(default_factory=list)
    
    @field_validator("score", "weight", "weighted_score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Score must be between 0 and 1")
        return v


class DQScore(BaseModel):
    """Overall data quality score."""
    
    dataset_id: UUID
    dataset_name: str
    overall_score: float
    dimensions: List[DimensionScore]
    grade: str
    passed_checks: int
    failed_checks: int
    total_checks: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator("overall_score")
    @classmethod
    def validate_overall_score(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Overall score must be between 0 and 1")
        return v
    
    @field_validator("grade")
    @classmethod
    def compute_grade(cls, v: str, values: Dict[str, Any]) -> str:
        score = values.get("overall_score", 0)
        if score >= 0.95:
            return "A+"
        elif score >= 0.90:
            return "A"
        elif score >= 0.85:
            return "B+"
        elif score >= 0.80:
            return "B"
        elif score >= 0.75:
            return "C+"
        elif score >= 0.70:
            return "C"
        elif score >= 0.60:
            return "D"
        else:
            return "F"


class RunResult(BaseModel):
    """Complete DQ scan run result."""
    
    id: UUID = Field(default_factory=uuid4)
    dataset: Dataset
    checks: List[CheckResult]
    profile: Optional[List[ProfileMetrics]] = None
    anomalies: Optional[List[AnomalyResult]] = None
    score: Optional[DQScore] = None
    started_at: datetime
    completed_at: datetime
    duration_ms: int
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class Signal(BaseModel):
    """Governance or quality signal."""
    
    name: str
    value: Any
    dimension: DQDimension
    weight: float = 1.0
    source: str = "system"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class Alert(BaseModel):
    """Alert notification."""
    
    id: UUID = Field(default_factory=uuid4)
    dataset_name: str
    severity: CheckSeverity
    title: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    channels: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    sent_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }