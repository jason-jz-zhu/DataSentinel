"""DQ Score computation with multi-dimensional weighting."""

from typing import Dict, List, Optional
import logging
from datetime import datetime

from dq_core.models import (
    CheckResult, DQScore, DimensionScore, DQDimension,
    CheckStatus, CheckSeverity, Signal
)
from dq_core.config import DQScoreWeights


logger = logging.getLogger(__name__)


class DQScorer:
    """Compute multi-dimensional data quality scores."""
    
    def __init__(self, weights: Optional[DQScoreWeights] = None):
        self.weights = weights or DQScoreWeights()
        self.weights.validate_sum()
    
    def compute_score(self, 
                     dataset_id: str,
                     dataset_name: str,
                     check_results: List[CheckResult],
                     signals: Optional[List[Signal]] = None) -> DQScore:
        """Compute overall DQ score from check results and signals."""
        signals = signals or []
        
        # Group checks by dimension
        dimension_checks = self._group_checks_by_dimension(check_results)
        
        # Compute dimension scores
        dimension_scores = []
        for dimension in DQDimension:
            checks = dimension_checks.get(dimension, [])
            dim_signals = [s for s in signals if s.dimension == dimension]
            
            score = self._compute_dimension_score(dimension, checks, dim_signals)
            dimension_scores.append(score)
        
        # Compute overall score
        overall_score = sum(
            dim_score.weighted_score for dim_score in dimension_scores
        )
        
        # Compute totals
        total_checks = len(check_results)
        passed_checks = len([c for c in check_results if c.status == CheckStatus.PASSED])
        failed_checks = total_checks - passed_checks
        
        # Determine grade
        grade = self._compute_grade(overall_score)
        
        return DQScore(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            overall_score=overall_score,
            dimensions=dimension_scores,
            grade=grade,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            total_checks=total_checks,
            timestamp=datetime.utcnow()
        )
    
    def _group_checks_by_dimension(self, 
                                  check_results: List[CheckResult]) -> Dict[DQDimension, List[CheckResult]]:
        """Group check results by DQ dimension."""
        grouped = {dim: [] for dim in DQDimension}
        
        for check in check_results:
            grouped[check.dimension].append(check)
        
        return grouped
    
    def _compute_dimension_score(self,
                                dimension: DQDimension,
                                checks: List[CheckResult],
                                signals: List[Signal]) -> DimensionScore:
        """Compute score for a single dimension."""
        # Get dimension weight
        weight = getattr(self.weights, dimension.value)
        
        # Compute base score from checks
        base_score = self._compute_checks_score(checks)
        
        # Apply signals adjustment
        signals_adjustment = self._compute_signals_adjustment(signals)
        
        # Final score (clamped to [0, 1])
        final_score = max(0.0, min(1.0, base_score + signals_adjustment))
        
        # Weighted score
        weighted_score = final_score * weight
        
        # Check counts
        passed = len([c for c in checks if c.status == CheckStatus.PASSED])
        failed = len(checks) - passed
        
        # Signal names
        signal_names = [s.name for s in signals]
        
        return DimensionScore(
            dimension=dimension,
            score=final_score,
            weight=weight,
            weighted_score=weighted_score,
            passed_checks=passed,
            failed_checks=failed,
            total_checks=len(checks),
            signals=signal_names
        )
    
    def _compute_checks_score(self, checks: List[CheckResult]) -> float:
        """Compute score from check results."""
        if not checks:
            return 1.0  # Perfect score if no checks
        
        # Weight checks by severity
        severity_weights = {
            CheckSeverity.CRITICAL: 1.0,
            CheckSeverity.HIGH: 0.8,
            CheckSeverity.MEDIUM: 0.6,
            CheckSeverity.LOW: 0.4,
            CheckSeverity.INFO: 0.2
        }
        
        total_weight = 0.0
        weighted_score = 0.0
        
        for check in checks:
            weight = severity_weights.get(check.severity, 0.6)
            total_weight += weight
            
            if check.status == CheckStatus.PASSED:
                check_score = check.pass_rate or 1.0
            elif check.status == CheckStatus.ERROR:
                check_score = 0.5  # Partial credit for errors
            else:
                check_score = check.pass_rate or 0.0
            
            weighted_score += check_score * weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _compute_signals_adjustment(self, signals: List[Signal]) -> float:
        """Compute adjustment from governance signals."""
        if not signals:
            return 0.0
        
        total_adjustment = 0.0
        
        for signal in signals:
            # Convert signal value to score adjustment
            if isinstance(signal.value, bool):
                adjustment = 0.1 if signal.value else -0.1
            elif isinstance(signal.value, (int, float)):
                # Normalize to [-0.2, 0.2] range
                adjustment = max(-0.2, min(0.2, float(signal.value) - 0.5))
            else:
                adjustment = 0.0
            
            total_adjustment += adjustment * signal.weight
        
        return total_adjustment
    
    def _compute_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 0.95:
            return "A+"
        elif score >= 0.90:
            return "A"
        elif score >= 0.85:
            return "B+"
        elif score >= 0.80:
            return "B"
        elif score >= 0.75:
            return "C+"
        elif score >= 0.70:
            return "C"
        elif score >= 0.60:
            return "D"
        else:
            return "F"
    
    def score_trends(self, historical_scores: List[DQScore]) -> Dict[str, float]:
        """Analyze score trends over time."""
        if len(historical_scores) < 2:
            return {"trend": 0.0, "volatility": 0.0}
        
        scores = [s.overall_score for s in historical_scores]
        
        # Calculate trend (linear regression slope)
        n = len(scores)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(scores) / n
        
        numerator = sum((x[i] - x_mean) * (scores[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        trend = numerator / denominator if denominator != 0 else 0.0
        
        # Calculate volatility (standard deviation)
        variance = sum((score - y_mean) ** 2 for score in scores) / n
        volatility = variance ** 0.5
        
        return {
            "trend": trend,
            "volatility": volatility,
            "latest_score": scores[-1],
            "average_score": y_mean,
            "min_score": min(scores),
            "max_score": max(scores)
        }
    
    def explain_score(self, dq_score: DQScore) -> Dict[str, str]:
        """Generate explanations for the DQ score."""
        explanations = {}
        
        # Overall explanation
        if dq_score.overall_score >= 0.9:
            explanations["overall"] = "Excellent data quality with minimal issues"
        elif dq_score.overall_score >= 0.8:
            explanations["overall"] = "Good data quality with minor issues"
        elif dq_score.overall_score >= 0.7:
            explanations["overall"] = "Acceptable data quality with some concerns"
        elif dq_score.overall_score >= 0.6:
            explanations["overall"] = "Poor data quality requiring attention"
        else:
            explanations["overall"] = "Critical data quality issues requiring immediate action"
        
        # Dimension explanations
        for dim_score in dq_score.dimensions:
            dim_name = dim_score.dimension.value
            
            if dim_score.failed_checks > 0:
                explanations[dim_name] = f"{dim_score.failed_checks} failed checks impacting {dim_name}"
            elif dim_score.score < 0.8:
                explanations[dim_name] = f"{dim_name} score below expectations"
            else:
                explanations[dim_name] = f"{dim_name} performing well"
        
        return explanations
    
    def recommend_actions(self, dq_score: DQScore) -> List[str]:
        """Recommend actions based on DQ score."""
        recommendations = []
        
        # Overall recommendations
        if dq_score.overall_score < 0.7:
            recommendations.append("Implement immediate data quality improvements")
        
        if dq_score.failed_checks > dq_score.total_checks * 0.2:
            recommendations.append("Review and fix failing data quality checks")
        
        # Dimension-specific recommendations
        for dim_score in dq_score.dimensions:
            if dim_score.score < 0.7:
                dim_name = dim_score.dimension.value
                
                if dim_name == "accuracy":
                    recommendations.append("Implement data validation rules and constraints")
                elif dim_name == "reliability":
                    recommendations.append("Monitor data freshness and pipeline SLAs")
                elif dim_name == "stewardship":
                    recommendations.append("Assign data owners and improve documentation")
                elif dim_name == "usability":
                    recommendations.append("Improve data schemas and accessibility")
        
        return recommendations