"""Snowflake adapter for SQL-based validation."""

from typing import Any, Dict, List, Optional
import logging
from datetime import datetime

import snowflake.connector
from snowflake.connector import DictCursor
from sqlalchemy import create_engine

from dq_core.registry import BaseAdapter, register_adapter
from dq_core.models import (
    Dataset, CheckResult, ProfileMetrics, CheckStatus,
    CheckSeverity, DQDimension
)
from dq_core.config import SnowflakeConfig


logger = logging.getLogger(__name__)


@register_adapter("snowflake")
class SnowflakeAdapter(BaseAdapter):
    """Adapter for Snowflake data warehouse."""
    
    def __init__(self, config: Optional[SnowflakeConfig] = None):
        self.config = config or SnowflakeConfig()
        self.connection = None
        self.engine = None
    
    def connect(self) -> None:
        """Connect to Snowflake."""
        if not all([self.config.account, self.config.user, self.config.password]):
            raise ValueError("Snowflake credentials not configured")
        
        conn_params = {
            "account": self.config.account,
            "user": self.config.user,
            "password": self.config.password,
            "warehouse": self.config.warehouse,
            "database": self.config.database,
            "schema": self.config.schema,
        }
        
        if self.config.role:
            conn_params["role"] = self.config.role
        
        self.connection = snowflake.connector.connect(**conn_params)
        
        # Create SQLAlchemy engine for advanced operations
        conn_string = (
            f"snowflake://{self.config.user}:{self.config.password}@"
            f"{self.config.account}/{self.config.database}/{self.config.schema}"
            f"?warehouse={self.config.warehouse}"
        )
        if self.config.role:
            conn_string += f"&role={self.config.role}"
        
        self.engine = create_engine(conn_string)
        logger.info(f"Connected to Snowflake: {self.config.account}")
    
    def validate(self, dataset: Dataset, checks: List[Any]) -> List[CheckResult]:
        """Run validation checks using SQL pushdown."""
        if not self.connection:
            self.connect()
        
        results = []
        table_name = dataset.location
        
        for check in checks:
            start_time = datetime.utcnow()
            try:
                result = self._run_check(table_name, check)
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
    
    def _run_check(self, table_name: str, check: Any) -> CheckResult:
        """Execute a single check via SQL."""
        start_time = datetime.utcnow()
        
        # Map check types to SQL generators
        sql_generators = {
            "not_null": self._sql_not_null,
            "unique": self._sql_unique,
            "range": self._sql_range,
            "values_in_set": self._sql_values_in_set,
            "regex": self._sql_regex,
            "row_count": self._sql_row_count,
            "freshness": self._sql_freshness,
            "foreign_key": self._sql_foreign_key,
        }
        
        sql_generator = sql_generators.get(check.type)
        if not sql_generator:
            raise ValueError(f"Unknown check type: {check.type}")
        
        sql = sql_generator(table_name, **check.parameters)
        
        cursor = self.connection.cursor(DictCursor)
        cursor.execute(sql)
        result = cursor.fetchone()
        cursor.close()
        
        passed = result.get("passed", 0)
        failed = result.get("failed", 0)
        total = result.get("total", passed + failed)
        
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
    
    def _sql_not_null(self, table: str, columns: List[str]) -> str:
        """Generate SQL for null check."""
        null_conditions = " OR ".join([f"{col} IS NULL" for col in columns])
        return f"""
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN NOT ({null_conditions}) THEN 1 END) as passed,
            COUNT(CASE WHEN {null_conditions} THEN 1 END) as failed
        FROM {table}
        """
    
    def _sql_unique(self, table: str, columns: List[str]) -> str:
        """Generate SQL for uniqueness check."""
        col_list = ", ".join(columns)
        return f"""
        WITH duplicates AS (
            SELECT {col_list}, COUNT(*) as cnt
            FROM {table}
            GROUP BY {col_list}
            HAVING COUNT(*) > 1
        ),
        stats AS (
            SELECT 
                COUNT(*) as total_rows,
                COALESCE(SUM(cnt), 0) as duplicate_rows
            FROM (
                SELECT COUNT(*) as cnt FROM {table}
                UNION ALL
                SELECT cnt FROM duplicates
            )
        )
        SELECT 
            total_rows as total,
            total_rows - duplicate_rows as passed,
            duplicate_rows as failed
        FROM stats
        """
    
    def _sql_range(self, table: str, column: str, 
                  min_value: Optional[float] = None,
                  max_value: Optional[float] = None) -> str:
        """Generate SQL for range check."""
        conditions = []
        if min_value is not None:
            conditions.append(f"{column} >= {min_value}")
        if max_value is not None:
            conditions.append(f"{column} <= {max_value}")
        
        condition = " AND ".join(conditions) if conditions else "TRUE"
        
        return f"""
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN {condition} THEN 1 END) as passed,
            COUNT(CASE WHEN NOT ({condition}) THEN 1 END) as failed
        FROM {table}
        """
    
    def _sql_values_in_set(self, table: str, column: str, 
                           allowed_values: List[Any]) -> str:
        """Generate SQL for set membership check."""
        values_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) 
                                for v in allowed_values])
        return f"""
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN {column} IN ({values_str}) THEN 1 END) as passed,
            COUNT(CASE WHEN {column} NOT IN ({values_str}) OR {column} IS NULL THEN 1 END) as failed
        FROM {table}
        """
    
    def _sql_regex(self, table: str, column: str, pattern: str) -> str:
        """Generate SQL for regex pattern check."""
        return f"""
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN REGEXP_LIKE({column}, '{pattern}') THEN 1 END) as passed,
            COUNT(CASE WHEN NOT REGEXP_LIKE({column}, '{pattern}') OR {column} IS NULL THEN 1 END) as failed
        FROM {table}
        """
    
    def _sql_row_count(self, table: str, 
                      min_rows: Optional[int] = None,
                      max_rows: Optional[int] = None) -> str:
        """Generate SQL for row count check."""
        return f"""
        WITH row_stats AS (
            SELECT COUNT(*) as cnt FROM {table}
        )
        SELECT 
            cnt as total,
            CASE 
                WHEN {min_rows or 0} <= cnt AND cnt <= {max_rows or 999999999} 
                THEN cnt 
                ELSE 0 
            END as passed,
            CASE 
                WHEN {min_rows or 0} > cnt OR cnt > {max_rows or 999999999} 
                THEN cnt 
                ELSE 0 
            END as failed
        FROM row_stats
        """
    
    def _sql_freshness(self, table: str, timestamp_column: str, 
                      max_age_hours: float) -> str:
        """Generate SQL for data freshness check."""
        return f"""
        WITH freshness AS (
            SELECT 
                MAX({timestamp_column}) as latest_timestamp,
                CURRENT_TIMESTAMP() as current_time,
                TIMESTAMPDIFF(HOUR, MAX({timestamp_column}), CURRENT_TIMESTAMP()) as age_hours
            FROM {table}
        )
        SELECT 
            1 as total,
            CASE WHEN age_hours <= {max_age_hours} THEN 1 ELSE 0 END as passed,
            CASE WHEN age_hours > {max_age_hours} THEN 1 ELSE 0 END as failed
        FROM freshness
        """
    
    def _sql_foreign_key(self, table: str, column: str, 
                         ref_table: str, ref_column: str) -> str:
        """Generate SQL for foreign key check."""
        return f"""
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN ref.{ref_column} IS NOT NULL THEN 1 END) as passed,
            COUNT(CASE WHEN ref.{ref_column} IS NULL AND t.{column} IS NOT NULL THEN 1 END) as failed
        FROM {table} t
        LEFT JOIN {ref_table} ref ON t.{column} = ref.{ref_column}
        """
    
    def profile(self, dataset: Dataset) -> List[ProfileMetrics]:
        """Profile the dataset using SQL queries."""
        if not self.connection:
            self.connect()
        
        table_name = dataset.location
        profiles = []
        
        # Get column information
        cursor = self.connection.cursor()
        cursor.execute(f"DESCRIBE TABLE {table_name}")
        columns = cursor.fetchall()
        
        # Get total row count
        cursor.execute(f"SELECT COUNT(*) as cnt FROM {table_name}")
        total_rows = cursor.fetchone()[0]
        
        for col_info in columns:
            col_name = col_info[0]
            col_type = col_info[1]
            
            # Basic statistics
            stats_sql = f"""
            SELECT 
                COUNT({col_name}) as count,
                COUNT(DISTINCT {col_name}) as distinct_count,
                SUM(CASE WHEN {col_name} IS NULL THEN 1 ELSE 0 END) as null_count
            FROM {table_name}
            """
            
            cursor.execute(stats_sql)
            stats = cursor.fetchone()
            
            null_percentage = (stats[2] / total_rows * 100) if total_rows > 0 else 0
            distinct_percentage = (stats[1] / total_rows * 100) if total_rows > 0 else 0
            
            profile = ProfileMetrics(
                column_name=col_name,
                data_type=col_type,
                count=stats[0],
                null_count=stats[2],
                null_percentage=null_percentage,
                distinct_count=stats[1],
                distinct_percentage=distinct_percentage
            )
            
            # Numeric statistics
            if any(t in col_type.upper() for t in ["INT", "FLOAT", "DOUBLE", "NUMBER", "DECIMAL"]):
                numeric_sql = f"""
                SELECT 
                    MIN({col_name}) as min_val,
                    MAX({col_name}) as max_val,
                    AVG({col_name}) as mean_val,
                    STDDEV({col_name}) as std_val,
                    MEDIAN({col_name}) as median_val,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {col_name}) as q25,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {col_name}) as q75
                FROM {table_name}
                """
                
                cursor.execute(numeric_sql)
                numeric_stats = cursor.fetchone()
                
                profile.min_value = float(numeric_stats[0]) if numeric_stats[0] else None
                profile.max_value = float(numeric_stats[1]) if numeric_stats[1] else None
                profile.mean = float(numeric_stats[2]) if numeric_stats[2] else None
                profile.std_dev = float(numeric_stats[3]) if numeric_stats[3] else None
                profile.median = float(numeric_stats[4]) if numeric_stats[4] else None
                
                if numeric_stats[5] and numeric_stats[6]:
                    profile.quantiles = {
                        "q25": float(numeric_stats[5]),
                        "q50": float(numeric_stats[4]),
                        "q75": float(numeric_stats[6])
                    }
            
            # String statistics
            elif "VARCHAR" in col_type.upper() or "TEXT" in col_type.upper():
                string_sql = f"""
                SELECT 
                    MIN(LENGTH({col_name})) as min_length,
                    MAX(LENGTH({col_name})) as max_length
                FROM {table_name}
                WHERE {col_name} IS NOT NULL
                """
                
                cursor.execute(string_sql)
                string_stats = cursor.fetchone()
                
                if string_stats:
                    profile.min_value = str(string_stats[0])
                    profile.max_value = str(string_stats[1])
                
                # Top values
                top_sql = f"""
                SELECT {col_name}, COUNT(*) as cnt
                FROM {table_name}
                GROUP BY {col_name}
                ORDER BY cnt DESC
                LIMIT 10
                """
                
                cursor.execute(top_sql)
                top_values = cursor.fetchall()
                
                profile.top_values = [
                    {"value": row[0], "count": row[1]}
                    for row in top_values
                ]
            
            profiles.append(profile)
        
        cursor.close()
        return profiles
    
    def close(self) -> None:
        """Close Snowflake connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
        if self.engine:
            self.engine.dispose()
            self.engine = None
        logger.info("Snowflake connection closed")