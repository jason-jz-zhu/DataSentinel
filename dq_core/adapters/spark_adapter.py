"""Spark adapter for DataFrame validation."""

from typing import Any, Dict, List, Optional
import logging
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

from dq_core.registry import BaseAdapter, register_adapter
from dq_core.models import (
    Dataset, CheckResult, ProfileMetrics, CheckStatus,
    CheckSeverity, DQDimension
)
from dq_core.config import SparkConfig


logger = logging.getLogger(__name__)


@register_adapter("spark")
class SparkAdapter(BaseAdapter):
    """Adapter for Apache Spark DataFrames."""
    
    def __init__(self, config: Optional[SparkConfig] = None):
        self.config = config or SparkConfig()
        self.spark: Optional[SparkSession] = None
    
    def connect(self) -> None:
        """Initialize Spark session."""
        builder = SparkSession.builder \
            .appName(self.config.app_name) \
            .master(self.config.master)
        
        # Apply configurations
        builder = builder.config("spark.sql.adaptive.enabled", 
                               str(self.config.adaptive_enabled).lower())
        builder = builder.config("spark.sql.shuffle.partitions", 
                               str(self.config.shuffle_partitions))
        builder = builder.config("spark.executor.memory", self.config.executor_memory)
        builder = builder.config("spark.driver.memory", self.config.driver_memory)
        
        for key, value in self.config.additional_configs.items():
            builder = builder.config(key, value)
        
        self.spark = builder.getOrCreate()
        logger.info(f"Spark session created: {self.spark.version}")
    
    def load_dataset(self, dataset: Dataset) -> DataFrame:
        """Load dataset as Spark DataFrame."""
        if not self.spark:
            self.connect()
        
        location = dataset.location
        
        # Determine file format from extension
        if location.endswith(".parquet"):
            df = self.spark.read.parquet(location)
        elif location.endswith(".csv"):
            df = self.spark.read.csv(location, header=True, inferSchema=True)
        elif location.endswith(".json"):
            df = self.spark.read.json(location)
        elif location.startswith("delta://"):
            df = self.spark.read.format("delta").load(location.replace("delta://", ""))
        else:
            # Assume it's a table name
            df = self.spark.table(location)
        
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
    
    def _run_check(self, df: DataFrame, check: Any) -> CheckResult:
        """Execute a single check."""
        start_time = datetime.utcnow()
        
        # Map check types to methods
        check_methods = {
            "not_null": self._check_not_null,
            "unique": self._check_unique,
            "range": self._check_range,
            "values_in_set": self._check_values_in_set,
            "regex": self._check_regex,
            "row_count": self._check_row_count,
        }
        
        method = check_methods.get(check.type)
        if not method:
            raise ValueError(f"Unknown check type: {check.type}")
        
        passed, failed, total = method(df, **check.parameters)
        
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
    
    def _check_not_null(self, df: DataFrame, columns: List[str]) -> tuple[int, int, int]:
        """Check for null values in specified columns."""
        total = df.count()
        
        null_condition = None
        for col in columns:
            if null_condition is None:
                null_condition = F.col(col).isNull()
            else:
                null_condition = null_condition | F.col(col).isNull()
        
        failed = df.filter(null_condition).count()
        passed = total - failed
        
        return passed, failed, total
    
    def _check_unique(self, df: DataFrame, columns: List[str]) -> tuple[int, int, int]:
        """Check for unique values in specified columns."""
        total = df.count()
        
        # Count duplicates
        duplicates = df.groupBy(columns).count() \
            .filter(F.col("count") > 1) \
            .agg(F.sum("count").alias("duplicate_count")) \
            .collect()
        
        failed = duplicates[0]["duplicate_count"] if duplicates else 0
        passed = total - failed
        
        return passed, failed, total
    
    def _check_range(self, df: DataFrame, column: str, 
                    min_value: Optional[float] = None, 
                    max_value: Optional[float] = None) -> tuple[int, int, int]:
        """Check if values are within specified range."""
        total = df.count()
        
        condition = F.lit(True)
        if min_value is not None:
            condition = condition & (F.col(column) >= min_value)
        if max_value is not None:
            condition = condition & (F.col(column) <= max_value)
        
        passed = df.filter(condition).count()
        failed = total - passed
        
        return passed, failed, total
    
    def _check_values_in_set(self, df: DataFrame, column: str, 
                             allowed_values: List[Any]) -> tuple[int, int, int]:
        """Check if values are in allowed set."""
        total = df.count()
        
        passed = df.filter(F.col(column).isin(allowed_values)).count()
        failed = total - passed
        
        return passed, failed, total
    
    def _check_regex(self, df: DataFrame, column: str, 
                    pattern: str) -> tuple[int, int, int]:
        """Check if values match regex pattern."""
        total = df.count()
        
        passed = df.filter(F.col(column).rlike(pattern)).count()
        failed = total - passed
        
        return passed, failed, total
    
    def _check_row_count(self, df: DataFrame, 
                        min_rows: Optional[int] = None,
                        max_rows: Optional[int] = None) -> tuple[int, int, int]:
        """Check if row count is within expected range."""
        total = df.count()
        
        passed = total
        failed = 0
        
        if min_rows is not None and total < min_rows:
            failed = min_rows - total
            passed = 0
        elif max_rows is not None and total > max_rows:
            failed = total - max_rows
            passed = max_rows
        
        return passed, failed, total
    
    def profile(self, dataset: Dataset) -> List[ProfileMetrics]:
        """Profile the dataset."""
        df = self.load_dataset(dataset)
        profiles = []
        
        total_rows = df.count()
        
        for field in df.schema.fields:
            col_name = field.name
            col_type = str(field.dataType)
            
            # Basic statistics
            stats = df.select(
                F.count(F.col(col_name)).alias("count"),
                F.count_distinct(F.col(col_name)).alias("distinct_count"),
                F.sum(F.when(F.col(col_name).isNull(), 1).otherwise(0)).alias("null_count")
            ).collect()[0]
            
            null_percentage = (stats["null_count"] / total_rows * 100) if total_rows > 0 else 0
            distinct_percentage = (stats["distinct_count"] / total_rows * 100) if total_rows > 0 else 0
            
            profile = ProfileMetrics(
                column_name=col_name,
                data_type=col_type,
                count=stats["count"],
                null_count=stats["null_count"],
                null_percentage=null_percentage,
                distinct_count=stats["distinct_count"],
                distinct_percentage=distinct_percentage
            )
            
            # Numeric statistics
            if field.dataType.simpleString() in ["int", "bigint", "float", "double", "decimal"]:
                numeric_stats = df.select(
                    F.min(col_name).alias("min"),
                    F.max(col_name).alias("max"),
                    F.mean(col_name).alias("mean"),
                    F.stddev(col_name).alias("std_dev")
                ).collect()[0]
                
                profile.min_value = numeric_stats["min"]
                profile.max_value = numeric_stats["max"]
                profile.mean = numeric_stats["mean"]
                profile.std_dev = numeric_stats["std_dev"]
                
                # Calculate quantiles
                quantiles = df.approxQuantile(col_name, [0.25, 0.5, 0.75], 0.01)
                if quantiles:
                    profile.quantiles = {
                        "q25": quantiles[0],
                        "q50": quantiles[1],
                        "q75": quantiles[2]
                    }
                    profile.median = quantiles[1]
            
            # String statistics
            elif field.dataType.simpleString() in ["string"]:
                string_stats = df.select(
                    F.min(F.length(col_name)).alias("min_length"),
                    F.max(F.length(col_name)).alias("max_length")
                ).collect()[0]
                
                profile.min_value = str(string_stats["min_length"])
                profile.max_value = str(string_stats["max_length"])
                
                # Top values
                top_values = df.groupBy(col_name).count() \
                    .orderBy(F.desc("count")) \
                    .limit(10) \
                    .collect()
                
                profile.top_values = [
                    {"value": row[col_name], "count": row["count"]}
                    for row in top_values
                ]
            
            profiles.append(profile)
        
        return profiles
    
    def close(self) -> None:
        """Close Spark session."""
        if self.spark:
            self.spark.stop()
            self.spark = None
            logger.info("Spark session closed")