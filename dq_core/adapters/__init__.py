"""Data adapters for different storage backends."""

from dq_core.adapters.spark_adapter import SparkAdapter
from dq_core.adapters.snowflake_adapter import SnowflakeAdapter
from dq_core.adapters.s3_adapter import S3Adapter
from dq_core.adapters.local_adapter import LocalAdapter

__all__ = ["SparkAdapter", "SnowflakeAdapter", "S3Adapter", "LocalAdapter"]