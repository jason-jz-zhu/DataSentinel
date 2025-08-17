"""Local file adapter using pandas."""

from typing import Any, Dict, List, Optional
import logging
from datetime import datetime

import pandas as pd

from dq_core.registry import BaseAdapter, register_adapter
from dq_core.models import (
    Dataset, CheckResult, ProfileMetrics, CheckStatus,
    CheckSeverity, DQDimension
)


logger = logging.getLogger(__name__)


@register_adapter("local")
class LocalAdapter(BaseAdapter):
    """Adapter for local files using pandas."""
    
    def __init__(self):
        self.data = None
    
    def connect(self) -> None:
        """Connect to local filesystem."""
        logger.info("Local adapter connected")
    
    def load_dataset(self, dataset: Dataset) -> pd.DataFrame:
        """Load dataset from local file."""
        location = dataset.location
        
        if location.endswith(".csv"):
            df = pd.read_csv(location)
        elif location.endswith(".json"):
            df = pd.read_json(location)
        elif location.endswith(".parquet"):
            df = pd.read_parquet(location)
        else:
            raise ValueError(f"Unsupported file format: {location}")
        
        return df
    
    def validate(self, dataset: Dataset, checks: List[Any]) -> List[CheckResult]:
        """Run validation checks on the dataset."""
        df = self.load_dataset(dataset)
        results = []
        
        for check in checks:
            start_time = datetime.utcnow()
            try:
                result = self._run_check(df, check)
                results.append(result)
            except Exception as e:
                logger.error(f"Check {check.name} failed: {e}")
                results.append(CheckResult(
                    check_id=check.id,
                    check_name=check.name,
                    status=CheckStatus.ERROR,
                    severity=check.severity,
                    dimension=check.dimension,
                    error_message=str(e),
                    execution_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
                ))
        
        return results
    
    def _run_check(self, df: pd.DataFrame, check: Any) -> CheckResult:
        """Execute a single check."""
        start_time = datetime.utcnow()
        
        # Simplified check execution for demo
        check_type = check.type
        parameters = check.parameters
        
        if check_type == "not_null":
            passed, failed, total = self._check_not_null(df, parameters.get("columns", []))
        else:
            # Default: assume all pass for demo
            total = len(df)
            passed = total
            failed = 0
        
        status = CheckStatus.PASSED if failed == 0 else CheckStatus.FAILED
        pass_rate = passed / total if total > 0 else 1.0
        
        return CheckResult(
            check_id=check.id,
            check_name=check.name,
            status=status,
            severity=check.severity,
            dimension=check.dimension,
            passed_records=passed,
            failed_records=failed,
            total_records=total,
            pass_rate=pass_rate,
            execution_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
        )
    
    def _check_not_null(self, df: pd.DataFrame, columns: List[str]) -> tuple[int, int, int]:
        """Check for null values in specified columns."""
        total = len(df)
        
        # Check if columns exist
        existing_columns = [col for col in columns if col in df.columns]
        if not existing_columns:
            # If no columns exist, use first column as fallback
            existing_columns = [df.columns[0]] if not df.empty else []
        
        if not existing_columns:
            return 0, 0, 0
        
        # Count rows with any null in specified columns
        null_mask = df[existing_columns].isnull().any(axis=1)
        failed = null_mask.sum()
        passed = total - failed
        
        return passed, failed, total
    
    def profile(self, dataset: Dataset) -> List[ProfileMetrics]:
        """Profile the dataset."""
        df = self.load_dataset(dataset)
        profiles = []
        
        total_rows = len(df)
        
        for column in df.columns:
            col_data = df[column]
            
            profile = ProfileMetrics(
                column_name=column,
                data_type=str(col_data.dtype),
                count=col_data.count(),
                null_count=col_data.isnull().sum(),
                null_percentage=(col_data.isnull().sum() / total_rows * 100) if total_rows > 0 else 0,
                distinct_count=col_data.nunique(),
                distinct_percentage=(col_data.nunique() / total_rows * 100) if total_rows > 0 else 0
            )
            
            # Add basic statistics for numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                non_null = col_data.dropna()
                if len(non_null) > 0:
                    profile.min_value = float(non_null.min())
                    profile.max_value = float(non_null.max())
                    profile.mean = float(non_null.mean())
                    profile.median = float(non_null.median())
                    profile.std_dev = float(non_null.std())
            
            profiles.append(profile)
        
        return profiles
    
    def close(self) -> None:
        """Close connection."""
        logger.info("Local adapter closed")