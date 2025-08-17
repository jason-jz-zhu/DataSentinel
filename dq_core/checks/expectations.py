"""Rule-based data quality expectations."""

from typing import Any, Dict, List, Optional, Union
from abc import abstractmethod
import re

from dq_core.registry import BaseCheck, register_check
from dq_core.models import CheckSeverity, DQDimension


class Expectation(BaseCheck):
    """Base class for expectations."""
    
    severity: CheckSeverity = CheckSeverity.MEDIUM
    dimension: DQDimension = DQDimension.ACCURACY
    
    @abstractmethod
    def run(self, data: Any, **params) -> Dict[str, Any]:
        """Run the expectation check."""
        pass
    
    def validate_params(self, **params) -> bool:
        """Validate expectation parameters."""
        return True


@register_check("not_null")
class NotNullCheck(Expectation):
    """Expect specified columns to not contain null values."""
    
    name = "not_null"
    
    def run(self, data: Any, columns: List[str], **params) -> Dict[str, Any]:
        """Check for null values in specified columns."""
        # Implementation handled by adapters
        return {
            "columns": columns,
            "check_type": "not_null"
        }
    
    def validate_params(self, columns: List[str], **params) -> bool:
        """Validate parameters."""
        if not columns or not isinstance(columns, list):
            raise ValueError("columns must be a non-empty list")
        return True


@register_check("unique")
class UniqueCheck(Expectation):
    """Expect specified columns to have unique values."""
    
    name = "unique"
    
    def run(self, data: Any, columns: List[str], **params) -> Dict[str, Any]:
        """Check for unique values in specified columns."""
        return {
            "columns": columns,
            "check_type": "unique"
        }
    
    def validate_params(self, columns: List[str], **params) -> bool:
        """Validate parameters."""
        if not columns or not isinstance(columns, list):
            raise ValueError("columns must be a non-empty list")
        return True


@register_check("range")
class RangeCheck(Expectation):
    """Expect column values to be within specified range."""
    
    name = "range"
    
    def run(self, data: Any, column: str, 
            min_value: Optional[Union[int, float]] = None,
            max_value: Optional[Union[int, float]] = None,
            **params) -> Dict[str, Any]:
        """Check if values are within range."""
        return {
            "column": column,
            "min_value": min_value,
            "max_value": max_value,
            "check_type": "range"
        }
    
    def validate_params(self, column: str, 
                       min_value: Optional[Union[int, float]] = None,
                       max_value: Optional[Union[int, float]] = None,
                       **params) -> bool:
        """Validate parameters."""
        if not column:
            raise ValueError("column is required")
        if min_value is None and max_value is None:
            raise ValueError("At least one of min_value or max_value must be specified")
        if min_value is not None and max_value is not None and min_value > max_value:
            raise ValueError("min_value must be less than or equal to max_value")
        return True


@register_check("values_in_set")
class ValuesInSetCheck(Expectation):
    """Expect column values to be in specified set."""
    
    name = "values_in_set"
    
    def run(self, data: Any, column: str, 
            allowed_values: List[Any], **params) -> Dict[str, Any]:
        """Check if values are in allowed set."""
        return {
            "column": column,
            "allowed_values": allowed_values,
            "check_type": "values_in_set"
        }
    
    def validate_params(self, column: str, 
                       allowed_values: List[Any], **params) -> bool:
        """Validate parameters."""
        if not column:
            raise ValueError("column is required")
        if not allowed_values or not isinstance(allowed_values, list):
            raise ValueError("allowed_values must be a non-empty list")
        return True


@register_check("regex")
class RegexCheck(Expectation):
    """Expect column values to match regex pattern."""
    
    name = "regex"
    
    def run(self, data: Any, column: str, 
            pattern: str, **params) -> Dict[str, Any]:
        """Check if values match regex pattern."""
        return {
            "column": column,
            "pattern": pattern,
            "check_type": "regex"
        }
    
    def validate_params(self, column: str, pattern: str, **params) -> bool:
        """Validate parameters."""
        if not column:
            raise ValueError("column is required")
        if not pattern:
            raise ValueError("pattern is required")
        try:
            re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
        return True


@register_check("foreign_key")
class ForeignKeyCheck(Expectation):
    """Expect foreign key relationship to be valid."""
    
    name = "foreign_key"
    dimension = DQDimension.ACCURACY
    
    def run(self, data: Any, 
            column: str,
            ref_table: str,
            ref_column: str,
            **params) -> Dict[str, Any]:
        """Check foreign key relationship."""
        return {
            "column": column,
            "ref_table": ref_table,
            "ref_column": ref_column,
            "check_type": "foreign_key"
        }
    
    def validate_params(self, column: str, 
                       ref_table: str,
                       ref_column: str, **params) -> bool:
        """Validate parameters."""
        if not all([column, ref_table, ref_column]):
            raise ValueError("column, ref_table, and ref_column are required")
        return True


@register_check("no_negative")
class NoNegativeCheck(Expectation):
    """Expect no negative values in numeric column."""
    
    name = "no_negative"
    
    def run(self, data: Any, column: str, **params) -> Dict[str, Any]:
        """Check for negative values."""
        return {
            "column": column,
            "min_value": 0,
            "check_type": "range"
        }
    
    def validate_params(self, column: str, **params) -> bool:
        """Validate parameters."""
        if not column:
            raise ValueError("column is required")
        return True


@register_check("monotonic_increasing")
class MonotonicIncreasingCheck(Expectation):
    """Expect column values to be monotonically increasing."""
    
    name = "monotonic_increasing"
    dimension = DQDimension.ACCURACY
    
    def run(self, data: Any, column: str, 
            group_by: Optional[List[str]] = None,
            **params) -> Dict[str, Any]:
        """Check if values are monotonically increasing."""
        return {
            "column": column,
            "group_by": group_by,
            "check_type": "monotonic_increasing"
        }
    
    def validate_params(self, column: str, 
                       group_by: Optional[List[str]] = None,
                       **params) -> bool:
        """Validate parameters."""
        if not column:
            raise ValueError("column is required")
        return True


@register_check("row_count")
class RowCountCheck(Expectation):
    """Expect row count to be within specified range."""
    
    name = "row_count"
    dimension = DQDimension.RELIABILITY
    
    def run(self, data: Any,
            min_rows: Optional[int] = None,
            max_rows: Optional[int] = None,
            **params) -> Dict[str, Any]:
        """Check row count."""
        return {
            "min_rows": min_rows,
            "max_rows": max_rows,
            "check_type": "row_count"
        }
    
    def validate_params(self,
                       min_rows: Optional[int] = None,
                       max_rows: Optional[int] = None,
                       **params) -> bool:
        """Validate parameters."""
        if min_rows is None and max_rows is None:
            raise ValueError("At least one of min_rows or max_rows must be specified")
        if min_rows is not None and min_rows < 0:
            raise ValueError("min_rows must be non-negative")
        if max_rows is not None and max_rows < 0:
            raise ValueError("max_rows must be non-negative")
        if min_rows is not None and max_rows is not None and min_rows > max_rows:
            raise ValueError("min_rows must be less than or equal to max_rows")
        return True


@register_check("column_exists")
class ColumnExistsCheck(Expectation):
    """Expect specified columns to exist in the dataset."""
    
    name = "column_exists"
    dimension = DQDimension.USABILITY
    
    def run(self, data: Any, columns: List[str], **params) -> Dict[str, Any]:
        """Check if columns exist."""
        return {
            "columns": columns,
            "check_type": "column_exists"
        }
    
    def validate_params(self, columns: List[str], **params) -> bool:
        """Validate parameters."""
        if not columns or not isinstance(columns, list):
            raise ValueError("columns must be a non-empty list")
        return True


@register_check("data_type")
class DataTypeCheck(Expectation):
    """Expect column to have specified data type."""
    
    name = "data_type"
    dimension = DQDimension.USABILITY
    
    def run(self, data: Any, column: str, 
            expected_type: str, **params) -> Dict[str, Any]:
        """Check column data type."""
        return {
            "column": column,
            "expected_type": expected_type,
            "check_type": "data_type"
        }
    
    def validate_params(self, column: str, 
                       expected_type: str, **params) -> bool:
        """Validate parameters."""
        if not column:
            raise ValueError("column is required")
        if not expected_type:
            raise ValueError("expected_type is required")
        return True


def create_expectation_suite(config: Dict[str, Any]) -> List[Expectation]:
    """Create a suite of expectations from configuration."""
    from dq_core.registry import check_registry
    
    expectations = []
    
    for check_config in config.get("checks", []):
        check_type = check_config.get("type")
        check_class = check_registry.get(check_type)
        
        if not check_class:
            raise ValueError(f"Unknown check type: {check_type}")
        
        check = check_class()
        
        # Validate parameters
        params = check_config.get("parameters", {})
        check.validate_params(**params)
        
        # Set metadata
        if "severity" in check_config:
            check.severity = CheckSeverity(check_config["severity"])
        if "dimension" in check_config:
            check.dimension = DQDimension(check_config["dimension"])
        
        expectations.append(check)
    
    return expectations