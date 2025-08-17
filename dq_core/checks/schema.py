"""Schema validation and drift detection checks."""

from typing import Any, Dict, List, Optional, Set
import json
import logging

from dq_core.registry import BaseCheck, register_check
from dq_core.models import CheckSeverity, DQDimension


logger = logging.getLogger(__name__)


@register_check("schema_match")
class SchemaCheck(BaseCheck):
    """Check if dataset schema matches expected schema."""
    
    name = "schema_match"
    severity = CheckSeverity.HIGH
    dimension = DQDimension.USABILITY
    
    def run(self, data: Any,
            expected_schema: Dict[str, str],
            allow_additional_columns: bool = False,
            **params) -> Dict[str, Any]:
        """Check schema match."""
        return {
            "expected_schema": expected_schema,
            "allow_additional_columns": allow_additional_columns,
            "check_type": "schema_match"
        }
    
    def validate_params(self,
                       expected_schema: Dict[str, str],
                       allow_additional_columns: bool = False,
                       **params) -> bool:
        """Validate parameters."""
        if not expected_schema:
            raise ValueError("expected_schema is required")
        if not isinstance(expected_schema, dict):
            raise ValueError("expected_schema must be a dictionary")
        return True


@register_check("schema_drift")
class SchemaDriftCheck(BaseCheck):
    """Detect schema changes from baseline."""
    
    name = "schema_drift"
    severity = CheckSeverity.MEDIUM
    dimension = DQDimension.USABILITY
    
    def run(self, data: Any,
            baseline_schema: Optional[Dict[str, str]] = None,
            allowed_changes: Optional[List[str]] = None,
            **params) -> Dict[str, Any]:
        """Check for schema drift."""
        return {
            "baseline_schema": baseline_schema,
            "allowed_changes": allowed_changes or [],
            "check_type": "schema_drift"
        }
    
    def validate_params(self,
                       baseline_schema: Optional[Dict[str, str]] = None,
                       allowed_changes: Optional[List[str]] = None,
                       **params) -> bool:
        """Validate parameters."""
        if baseline_schema and not isinstance(baseline_schema, dict):
            raise ValueError("baseline_schema must be a dictionary")
        if allowed_changes and not isinstance(allowed_changes, list):
            raise ValueError("allowed_changes must be a list")
        return True


@register_check("column_order")
class ColumnOrderCheck(BaseCheck):
    """Check if columns appear in expected order."""
    
    name = "column_order"
    severity = CheckSeverity.LOW
    dimension = DQDimension.USABILITY
    
    def run(self, data: Any,
            expected_order: List[str],
            strict: bool = False,
            **params) -> Dict[str, Any]:
        """Check column order."""
        return {
            "expected_order": expected_order,
            "strict": strict,
            "check_type": "column_order"
        }
    
    def validate_params(self,
                       expected_order: List[str],
                       strict: bool = False,
                       **params) -> bool:
        """Validate parameters."""
        if not expected_order:
            raise ValueError("expected_order is required")
        if not isinstance(expected_order, list):
            raise ValueError("expected_order must be a list")
        return True


@register_check("required_columns")
class RequiredColumnsCheck(BaseCheck):
    """Check if all required columns are present."""
    
    name = "required_columns"
    severity = CheckSeverity.HIGH
    dimension = DQDimension.USABILITY
    
    def run(self, data: Any,
            required_columns: List[str],
            **params) -> Dict[str, Any]:
        """Check for required columns."""
        return {
            "required_columns": required_columns,
            "check_type": "required_columns"
        }
    
    def validate_params(self,
                       required_columns: List[str],
                       **params) -> bool:
        """Validate parameters."""
        if not required_columns:
            raise ValueError("required_columns is required")
        if not isinstance(required_columns, list):
            raise ValueError("required_columns must be a list")
        return True


@register_check("column_types")
class ColumnTypesCheck(BaseCheck):
    """Check if column data types match expectations."""
    
    name = "column_types"
    severity = CheckSeverity.MEDIUM
    dimension = DQDimension.USABILITY
    
    def run(self, data: Any,
            expected_types: Dict[str, str],
            type_mapping: Optional[Dict[str, List[str]]] = None,
            **params) -> Dict[str, Any]:
        """Check column types."""
        # Default type mappings for common variations
        default_mapping = {
            "integer": ["int", "bigint", "integer", "int32", "int64"],
            "float": ["float", "double", "decimal", "numeric", "float32", "float64"],
            "string": ["string", "varchar", "text", "char"],
            "boolean": ["boolean", "bool"],
            "date": ["date"],
            "timestamp": ["timestamp", "datetime", "timestamp_ntz", "timestamp_tz"],
        }
        
        type_mapping = type_mapping or default_mapping
        
        return {
            "expected_types": expected_types,
            "type_mapping": type_mapping,
            "check_type": "column_types"
        }
    
    def validate_params(self,
                       expected_types: Dict[str, str],
                       type_mapping: Optional[Dict[str, List[str]]] = None,
                       **params) -> bool:
        """Validate parameters."""
        if not expected_types:
            raise ValueError("expected_types is required")
        if not isinstance(expected_types, dict):
            raise ValueError("expected_types must be a dictionary")
        if type_mapping and not isinstance(type_mapping, dict):
            raise ValueError("type_mapping must be a dictionary")
        return True


@register_check("schema_compatibility")
class SchemaCompatibilityCheck(BaseCheck):
    """Check if schema is compatible with downstream systems."""
    
    name = "schema_compatibility"
    severity = CheckSeverity.HIGH
    dimension = DQDimension.USABILITY
    
    def run(self, data: Any,
            compatibility_rules: Dict[str, Any],
            **params) -> Dict[str, Any]:
        """Check schema compatibility."""
        return {
            "compatibility_rules": compatibility_rules,
            "check_type": "schema_compatibility"
        }
    
    def validate_params(self,
                       compatibility_rules: Dict[str, Any],
                       **params) -> bool:
        """Validate parameters."""
        if not compatibility_rules:
            raise ValueError("compatibility_rules is required")
        if not isinstance(compatibility_rules, dict):
            raise ValueError("compatibility_rules must be a dictionary")
        return True


class SchemaRegistry:
    """Registry for storing and retrieving schemas."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or "./schemas"
        self.schemas: Dict[str, Dict[str, Any]] = {}
    
    def register(self, dataset_name: str, schema: Dict[str, Any]) -> None:
        """Register a schema for a dataset."""
        from datetime import datetime
        
        self.schemas[dataset_name] = {
            "schema": schema,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if self.storage_path:
            self._save_schema(dataset_name, schema)
    
    def get(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a dataset."""
        if dataset_name in self.schemas:
            return self.schemas[dataset_name]["schema"]
        
        if self.storage_path:
            return self._load_schema(dataset_name)
        
        return None
    
    def _save_schema(self, dataset_name: str, schema: Dict[str, Any]) -> None:
        """Save schema to file."""
        import os
        os.makedirs(self.storage_path, exist_ok=True)
        
        file_path = os.path.join(self.storage_path, f"{dataset_name}.json")
        with open(file_path, "w") as f:
            json.dump(schema, f, indent=2)
    
    def _load_schema(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Load schema from file."""
        import os
        file_path = os.path.join(self.storage_path, f"{dataset_name}.json")
        
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        
        return None
    
    def detect_drift(self, dataset_name: str, 
                    current_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Detect schema drift from baseline."""
        baseline = self.get(dataset_name)
        
        if not baseline:
            return {
                "has_drift": False,
                "message": "No baseline schema found"
            }
        
        baseline_cols = set(baseline.keys())
        current_cols = set(current_schema.keys())
        
        added_cols = current_cols - baseline_cols
        removed_cols = baseline_cols - current_cols
        
        type_changes = {}
        for col in baseline_cols & current_cols:
            if baseline[col] != current_schema[col]:
                type_changes[col] = {
                    "from": baseline[col],
                    "to": current_schema[col]
                }
        
        has_drift = bool(added_cols or removed_cols or type_changes)
        
        return {
            "has_drift": has_drift,
            "added_columns": list(added_cols),
            "removed_columns": list(removed_cols),
            "type_changes": type_changes
        }