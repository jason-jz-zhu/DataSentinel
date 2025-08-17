"""Data freshness and timeliness checks."""

from typing import Any, Dict, Optional
from datetime import datetime, timedelta
import logging

from dq_core.registry import BaseCheck, register_check
from dq_core.models import CheckSeverity, DQDimension


logger = logging.getLogger(__name__)


@register_check("freshness")
class FreshnessCheck(BaseCheck):
    """Check data freshness based on timestamp columns."""
    
    name = "freshness"
    severity = CheckSeverity.HIGH
    dimension = DQDimension.RELIABILITY
    
    def run(self, data: Any,
            timestamp_column: str,
            max_age_hours: float,
            **params) -> Dict[str, Any]:
        """Check if data is fresh enough."""
        return {
            "timestamp_column": timestamp_column,
            "max_age_hours": max_age_hours,
            "check_type": "freshness"
        }
    
    def validate_params(self,
                       timestamp_column: str,
                       max_age_hours: float,
                       **params) -> bool:
        """Validate parameters."""
        if not timestamp_column:
            raise ValueError("timestamp_column is required")
        if max_age_hours <= 0:
            raise ValueError("max_age_hours must be positive")
        return True


@register_check("landing_time")
class LandingTimeCheck(BaseCheck):
    """Check if data landed within expected SLA window."""
    
    name = "landing_time"
    severity = CheckSeverity.HIGH
    dimension = DQDimension.RELIABILITY
    
    def run(self, data: Any,
            expected_hour: int,
            tolerance_minutes: int = 30,
            partition_column: Optional[str] = None,
            **params) -> Dict[str, Any]:
        """Check if data landed on time."""
        current_time = datetime.now()
        expected_time = current_time.replace(
            hour=expected_hour,
            minute=0,
            second=0,
            microsecond=0
        )
        
        # If expected time is in the future, use yesterday
        if expected_time > current_time:
            expected_time -= timedelta(days=1)
        
        return {
            "expected_time": expected_time.isoformat(),
            "tolerance_minutes": tolerance_minutes,
            "partition_column": partition_column,
            "check_type": "landing_time"
        }
    
    def validate_params(self,
                       expected_hour: int,
                       tolerance_minutes: int = 30,
                       **params) -> bool:
        """Validate parameters."""
        if not 0 <= expected_hour <= 23:
            raise ValueError("expected_hour must be between 0 and 23")
        if tolerance_minutes < 0:
            raise ValueError("tolerance_minutes must be non-negative")
        return True


@register_check("partition_completeness")
class PartitionCompletenessCheck(BaseCheck):
    """Check if all expected partitions are present."""
    
    name = "partition_completeness"
    severity = CheckSeverity.HIGH
    dimension = DQDimension.RELIABILITY
    
    def run(self, data: Any,
            partition_column: str,
            expected_partitions: int,
            lookback_days: int = 7,
            **params) -> Dict[str, Any]:
        """Check partition completeness."""
        return {
            "partition_column": partition_column,
            "expected_partitions": expected_partitions,
            "lookback_days": lookback_days,
            "check_type": "partition_completeness"
        }
    
    def validate_params(self,
                       partition_column: str,
                       expected_partitions: int,
                       lookback_days: int = 7,
                       **params) -> bool:
        """Validate parameters."""
        if not partition_column:
            raise ValueError("partition_column is required")
        if expected_partitions <= 0:
            raise ValueError("expected_partitions must be positive")
        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        return True


@register_check("update_frequency")
class UpdateFrequencyCheck(BaseCheck):
    """Check if data is being updated at expected frequency."""
    
    name = "update_frequency"
    severity = CheckSeverity.MEDIUM
    dimension = DQDimension.RELIABILITY
    
    def run(self, data: Any,
            timestamp_column: str,
            expected_frequency_hours: float,
            min_updates: int = 1,
            lookback_hours: float = 24,
            **params) -> Dict[str, Any]:
        """Check update frequency."""
        return {
            "timestamp_column": timestamp_column,
            "expected_frequency_hours": expected_frequency_hours,
            "min_updates": min_updates,
            "lookback_hours": lookback_hours,
            "check_type": "update_frequency"
        }
    
    def validate_params(self,
                       timestamp_column: str,
                       expected_frequency_hours: float,
                       min_updates: int = 1,
                       lookback_hours: float = 24,
                       **params) -> bool:
        """Validate parameters."""
        if not timestamp_column:
            raise ValueError("timestamp_column is required")
        if expected_frequency_hours <= 0:
            raise ValueError("expected_frequency_hours must be positive")
        if min_updates <= 0:
            raise ValueError("min_updates must be positive")
        if lookback_hours <= 0:
            raise ValueError("lookback_hours must be positive")
        return True


@register_check("data_delay")
class DataDelayCheck(BaseCheck):
    """Check for data processing delays."""
    
    name = "data_delay"
    severity = CheckSeverity.HIGH
    dimension = DQDimension.RELIABILITY
    
    def run(self, data: Any,
            event_time_column: str,
            process_time_column: str,
            max_delay_hours: float,
            **params) -> Dict[str, Any]:
        """Check data processing delay."""
        return {
            "event_time_column": event_time_column,
            "process_time_column": process_time_column,
            "max_delay_hours": max_delay_hours,
            "check_type": "data_delay"
        }
    
    def validate_params(self,
                       event_time_column: str,
                       process_time_column: str,
                       max_delay_hours: float,
                       **params) -> bool:
        """Validate parameters."""
        if not event_time_column:
            raise ValueError("event_time_column is required")
        if not process_time_column:
            raise ValueError("process_time_column is required")
        if max_delay_hours <= 0:
            raise ValueError("max_delay_hours must be positive")
        return True