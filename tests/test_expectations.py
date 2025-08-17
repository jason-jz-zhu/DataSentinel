"""Tests for data quality expectations."""

import pytest
import pandas as pd
from dq_core.checks.expectations import (
    NotNullCheck, UniqueCheck, RangeCheck, 
    ValuesInSetCheck, RegexCheck
)


class TestNotNullCheck:
    """Test not null expectations."""
    
    def test_validate_params_valid(self):
        check = NotNullCheck()
        assert check.validate_params(columns=["id", "name"])
    
    def test_validate_params_empty_columns(self):
        check = NotNullCheck()
        with pytest.raises(ValueError):
            check.validate_params(columns=[])
    
    def test_run_returns_config(self):
        check = NotNullCheck()
        result = check.run(None, columns=["id"])
        assert result["columns"] == ["id"]
        assert result["check_type"] == "not_null"


class TestUniqueCheck:
    """Test unique value expectations."""
    
    def test_validate_params_valid(self):
        check = UniqueCheck()
        assert check.validate_params(columns=["email"])
    
    def test_run_returns_config(self):
        check = UniqueCheck()
        result = check.run(None, columns=["email"])
        assert result["columns"] == ["email"]
        assert result["check_type"] == "unique"


class TestRangeCheck:
    """Test range validation expectations."""
    
    def test_validate_params_valid(self):
        check = RangeCheck()
        assert check.validate_params(column="age", min_value=0, max_value=120)
    
    def test_validate_params_no_bounds(self):
        check = RangeCheck()
        with pytest.raises(ValueError):
            check.validate_params(column="age")
    
    def test_validate_params_invalid_range(self):
        check = RangeCheck()
        with pytest.raises(ValueError):
            check.validate_params(column="age", min_value=120, max_value=0)
    
    def test_run_returns_config(self):
        check = RangeCheck()
        result = check.run(None, column="age", min_value=0, max_value=120)
        assert result["column"] == "age"
        assert result["min_value"] == 0
        assert result["max_value"] == 120


class TestValuesInSetCheck:
    """Test set membership expectations."""
    
    def test_validate_params_valid(self):
        check = ValuesInSetCheck()
        assert check.validate_params(column="status", allowed_values=["active", "inactive"])
    
    def test_validate_params_empty_set(self):
        check = ValuesInSetCheck()
        with pytest.raises(ValueError):
            check.validate_params(column="status", allowed_values=[])
    
    def test_run_returns_config(self):
        check = ValuesInSetCheck()
        result = check.run(None, column="status", allowed_values=["active"])
        assert result["column"] == "status"
        assert result["allowed_values"] == ["active"]


class TestRegexCheck:
    """Test regex pattern expectations."""
    
    def test_validate_params_valid(self):
        check = RegexCheck()
        assert check.validate_params(column="email", pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    
    def test_validate_params_invalid_regex(self):
        check = RegexCheck()
        with pytest.raises(ValueError):
            check.validate_params(column="email", pattern="[invalid")
    
    def test_run_returns_config(self):
        check = RegexCheck()
        pattern = r"^\d{3}-\d{2}-\d{4}$"
        result = check.run(None, column="ssn", pattern=pattern)
        assert result["column"] == "ssn"
        assert result["pattern"] == pattern