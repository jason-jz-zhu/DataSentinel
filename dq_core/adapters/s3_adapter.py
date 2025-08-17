"""S3 adapter for file-based validation."""

from typing import Any, Dict, List, Optional
import logging
from datetime import datetime
import io
import json

import boto3
import pandas as pd
import s3fs
from botocore.exceptions import ClientError

from dq_core.registry import BaseAdapter, register_adapter
from dq_core.models import (
    Dataset, CheckResult, ProfileMetrics, CheckStatus
)
from dq_core.config import S3Config
from dq_core.adapters.spark_adapter import SparkAdapter


logger = logging.getLogger(__name__)


@register_adapter("s3")
class S3Adapter(BaseAdapter):
    """Adapter for S3/MinIO file storage."""
    
    def __init__(self, config: Optional[S3Config] = None):
        self.config = config or S3Config()
        self.s3_client = None
        self.fs = None
        self.spark_adapter = None
    
    def connect(self) -> None:
        """Connect to S3/MinIO."""
        # Create boto3 client
        client_kwargs = {
            "region_name": self.config.region,
            "use_ssl": self.config.use_ssl
        }
        
        if self.config.endpoint_url:
            client_kwargs["endpoint_url"] = self.config.endpoint_url
        
        if self.config.access_key and self.config.secret_key:
            client_kwargs["aws_access_key_id"] = self.config.access_key
            client_kwargs["aws_secret_access_key"] = self.config.secret_key
        
        self.s3_client = boto3.client("s3", **client_kwargs)
        
        # Create s3fs filesystem
        fs_kwargs = {
            "anon": False,
            "use_ssl": self.config.use_ssl
        }
        
        if self.config.endpoint_url:
            fs_kwargs["client_kwargs"] = {"endpoint_url": self.config.endpoint_url}
        
        if self.config.access_key and self.config.secret_key:
            fs_kwargs["key"] = self.config.access_key
            fs_kwargs["secret"] = self.config.secret_key
        
        self.fs = s3fs.S3FileSystem(**fs_kwargs)
        
        # Initialize Spark adapter for complex processing
        self.spark_adapter = SparkAdapter()
        
        logger.info(f"Connected to S3: {self.config.endpoint_url or 'AWS'}")
    
    def _parse_s3_path(self, path: str) -> tuple[str, str]:
        """Parse S3 path into bucket and key."""
        if path.startswith("s3://"):
            path = path[5:]
        elif path.startswith("s3a://"):
            path = path[6:]
        
        parts = path.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        
        return bucket, key
    
    def _read_file(self, path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Read file from S3 into pandas DataFrame."""
        if not self.fs:
            self.connect()
        
        # Determine file format
        if path.endswith(".parquet"):
            df = pd.read_parquet(f"s3://{path}", filesystem=self.fs)
        elif path.endswith(".csv"):
            df = pd.read_csv(f"s3://{path}", storage_options={"fs": self.fs})
        elif path.endswith(".json") or path.endswith(".jsonl"):
            with self.fs.open(path, "r") as f:
                if path.endswith(".jsonl"):
                    lines = f.readlines()
                    data = [json.loads(line) for line in lines]
                    df = pd.DataFrame(data)
                else:
                    df = pd.read_json(f)
        else:
            raise ValueError(f"Unsupported file format for {path}")
        
        # Apply sampling if requested
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        return df
    
    def _list_files(self, prefix: str) -> List[str]:
        """List files in S3 prefix."""
        bucket, key = self._parse_s3_path(prefix)
        
        paginator = self.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=key)
        
        files = []
        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    files.append(f"{bucket}/{obj['Key']}")
        
        return files
    
    def validate(self, dataset: Dataset, checks: List[Any]) -> List[CheckResult]:
        """Run validation checks on S3 data."""
        location = dataset.location
        
        # Check if location is a single file or directory
        if location.endswith("/"):
            # Directory - process multiple files
            files = self._list_files(location)
            if not files:
                raise ValueError(f"No files found in {location}")
            
            # For large datasets, use Spark
            if len(files) > 10 or any(self._get_file_size(f) > 100_000_000 for f in files):
                return self._validate_with_spark(dataset, checks)
            
            # For small datasets, use pandas
            dfs = []
            for file in files[:10]:  # Limit to first 10 files
                try:
                    df = self._read_file(file, sample_size=10000)
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"Failed to read {file}: {e}")
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                return self._validate_pandas(combined_df, checks, dataset)
        else:
            # Single file
            file_size = self._get_file_size(location)
            
            # Use Spark for large files
            if file_size > 100_000_000:  # 100MB
                return self._validate_with_spark(dataset, checks)
            
            # Use pandas for small files
            df = self._read_file(location)
            return self._validate_pandas(df, checks, dataset)
        
        return []
    
    def _get_file_size(self, path: str) -> int:
        """Get file size in bytes."""
        bucket, key = self._parse_s3_path(path)
        try:
            response = self.s3_client.head_object(Bucket=bucket, Key=key)
            return response["ContentLength"]
        except ClientError:
            return 0
    
    def _validate_with_spark(self, dataset: Dataset, checks: List[Any]) -> List[CheckResult]:
        """Validate using Spark for large datasets."""
        if not self.spark_adapter:
            self.spark_adapter = SparkAdapter()
        
        # Modify dataset location to use S3A protocol for Spark
        spark_dataset = Dataset(**dataset.model_dump())
        if spark_dataset.location.startswith("s3://"):
            spark_dataset.location = spark_dataset.location.replace("s3://", "s3a://")
        
        return self.spark_adapter.validate(spark_dataset, checks)
    
    def _validate_pandas(self, df: pd.DataFrame, checks: List[Any], 
                        dataset: Dataset) -> List[CheckResult]:
        """Validate using pandas for small datasets."""
        results = []
        
        for check in checks:
            start_time = datetime.utcnow()
            try:
                result = self._run_pandas_check(df, check)
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
    
    def _run_pandas_check(self, df: pd.DataFrame, check: Any) -> CheckResult:
        """Execute check on pandas DataFrame."""
        start_time = datetime.utcnow()
        
        check_methods = {
            "not_null": self._check_not_null_pandas,
            "unique": self._check_unique_pandas,
            "range": self._check_range_pandas,
            "values_in_set": self._check_values_in_set_pandas,
            "regex": self._check_regex_pandas,
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
    
    def _check_not_null_pandas(self, df: pd.DataFrame, 
                               columns: List[str]) -> tuple[int, int, int]:
        """Check for null values in pandas DataFrame."""
        total = len(df)
        null_mask = df[columns].isnull().any(axis=1)
        failed = null_mask.sum()
        passed = total - failed
        return passed, failed, total
    
    def _check_unique_pandas(self, df: pd.DataFrame, 
                             columns: List[str]) -> tuple[int, int, int]:
        """Check for unique values in pandas DataFrame."""
        total = len(df)
        duplicates = df.duplicated(subset=columns, keep=False)
        failed = duplicates.sum()
        passed = total - failed
        return passed, failed, total
    
    def _check_range_pandas(self, df: pd.DataFrame, column: str,
                           min_value: Optional[float] = None,
                           max_value: Optional[float] = None) -> tuple[int, int, int]:
        """Check if values are within range in pandas DataFrame."""
        total = len(df)
        mask = pd.Series([True] * total)
        
        if min_value is not None:
            mask &= df[column] >= min_value
        if max_value is not None:
            mask &= df[column] <= max_value
        
        passed = mask.sum()
        failed = total - passed
        return passed, failed, total
    
    def _check_values_in_set_pandas(self, df: pd.DataFrame, column: str,
                                   allowed_values: List[Any]) -> tuple[int, int, int]:
        """Check if values are in allowed set in pandas DataFrame."""
        total = len(df)
        mask = df[column].isin(allowed_values)
        passed = mask.sum()
        failed = total - passed
        return passed, failed, total
    
    def _check_regex_pandas(self, df: pd.DataFrame, column: str,
                           pattern: str) -> tuple[int, int, int]:
        """Check if values match regex in pandas DataFrame."""
        total = len(df)
        mask = df[column].astype(str).str.match(pattern)
        passed = mask.sum()
        failed = total - passed
        return passed, failed, total
    
    def profile(self, dataset: Dataset) -> List[ProfileMetrics]:
        """Profile S3 dataset."""
        location = dataset.location
        
        # For large datasets, use Spark
        file_size = self._get_file_size(location) if not location.endswith("/") else 0
        
        if file_size > 100_000_000 or location.endswith("/"):
            if not self.spark_adapter:
                self.spark_adapter = SparkAdapter()
            
            spark_dataset = Dataset(**dataset.model_dump())
            if spark_dataset.location.startswith("s3://"):
                spark_dataset.location = spark_dataset.location.replace("s3://", "s3a://")
            
            return self.spark_adapter.profile(spark_dataset)
        
        # For small files, use pandas
        df = self._read_file(location)
        return self._profile_pandas(df)
    
    def _profile_pandas(self, df: pd.DataFrame) -> List[ProfileMetrics]:
        """Profile pandas DataFrame."""
        profiles = []
        total_rows = len(df)
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            
            profile = ProfileMetrics(
                column_name=col,
                data_type=col_type,
                count=df[col].count(),
                null_count=df[col].isnull().sum(),
                null_percentage=(df[col].isnull().sum() / total_rows * 100) if total_rows > 0 else 0,
                distinct_count=df[col].nunique(),
                distinct_percentage=(df[col].nunique() / total_rows * 100) if total_rows > 0 else 0
            )
            
            # Numeric statistics
            if pd.api.types.is_numeric_dtype(df[col]):
                profile.min_value = float(df[col].min())
                profile.max_value = float(df[col].max())
                profile.mean = float(df[col].mean())
                profile.median = float(df[col].median())
                profile.std_dev = float(df[col].std())
                
                quantiles = df[col].quantile([0.25, 0.5, 0.75])
                profile.quantiles = {
                    "q25": float(quantiles[0.25]),
                    "q50": float(quantiles[0.5]),
                    "q75": float(quantiles[0.75])
                }
            
            # String statistics
            elif pd.api.types.is_string_dtype(df[col]):
                str_lengths = df[col].astype(str).str.len()
                profile.min_value = str(str_lengths.min())
                profile.max_value = str(str_lengths.max())
                
                # Top values
                top_values = df[col].value_counts().head(10)
                profile.top_values = [
                    {"value": value, "count": int(count)}
                    for value, count in top_values.items()
                ]
            
            profiles.append(profile)
        
        return profiles
    
    def close(self) -> None:
        """Close connections."""
        if self.spark_adapter:
            self.spark_adapter.close()
            self.spark_adapter = None
        
        self.s3_client = None
        self.fs = None
        logger.info("S3 adapter closed")