"""Tests for DQ scoring system."""

import pytest
from datetime import datetime
from dq_core.scoring.score import DQScorer
from dq_core.models import (
    CheckResult, CheckStatus, CheckSeverity, DQDimension
)
from dq_core.config import DQScoreWeights


class TestDQScorer:
    """Test DQ scoring functionality."""
    
    def test_compute_score_all_passed(self):
        scorer = DQScorer()
        
        check_results = [
            CheckResult(
                check_id="1",
                check_name="test_check_1",
                status=CheckStatus.PASSED,
                severity=CheckSeverity.HIGH,
                dimension=DQDimension.ACCURACY,
                pass_rate=1.0,
                passed_records=100,
                failed_records=0,
                total_records=100
            ),
            CheckResult(
                check_id="2",
                check_name="test_check_2",
                status=CheckStatus.PASSED,
                severity=CheckSeverity.MEDIUM,
                dimension=DQDimension.RELIABILITY,
                pass_rate=1.0,
                passed_records=100,
                failed_records=0,
                total_records=100
            )
        ]
        
        score = scorer.compute_score("dataset_1", "test_dataset", check_results)
        
        assert score.overall_score == 1.0
        assert score.grade == "A+"
        assert score.passed_checks == 2
        assert score.failed_checks == 0
        assert score.total_checks == 2
    
    def test_compute_score_some_failed(self):
        scorer = DQScorer()
        
        check_results = [
            CheckResult(
                check_id="1",
                check_name="test_check_1",
                status=CheckStatus.PASSED,
                severity=CheckSeverity.HIGH,
                dimension=DQDimension.ACCURACY,
                pass_rate=1.0
            ),
            CheckResult(
                check_id="2",
                check_name="test_check_2",
                status=CheckStatus.FAILED,
                severity=CheckSeverity.HIGH,
                dimension=DQDimension.ACCURACY,
                pass_rate=0.5
            )
        ]
        
        score = scorer.compute_score("dataset_1", "test_dataset", check_results)
        
        assert score.overall_score < 1.0
        assert score.passed_checks == 1
        assert score.failed_checks == 1
    
    def test_grade_computation(self):
        scorer = DQScorer()
        
        assert scorer._compute_grade(0.95) == "A+"
        assert scorer._compute_grade(0.90) == "A"
        assert scorer._compute_grade(0.85) == "B+"
        assert scorer._compute_grade(0.80) == "B"
        assert scorer._compute_grade(0.75) == "C+"
        assert scorer._compute_grade(0.70) == "C"
        assert scorer._compute_grade(0.60) == "D"
        assert scorer._compute_grade(0.50) == "F"
    
    def test_custom_weights(self):
        custom_weights = DQScoreWeights(
            accuracy=0.5,
            reliability=0.3,
            stewardship=0.1,
            usability=0.1
        )
        
        scorer = DQScorer(custom_weights)
        assert scorer.weights.accuracy == 0.5
    
    def test_invalid_weights_sum(self):
        with pytest.raises(ValueError):
            weights = DQScoreWeights(
                accuracy=0.5,
                reliability=0.5,
                stewardship=0.5,
                usability=0.5
            )
            weights.validate_sum()