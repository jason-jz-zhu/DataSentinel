"""Data profiling with WhyLogs/Evidently integration."""

from typing import Any, Dict, List, Optional, Union
import logging
from datetime import datetime
import json

import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

try:
    import whylogs as why
    from whylogs.core import DatasetProfileView
    WHYLOGS_AVAILABLE = True
except ImportError:
    WHYLOGS_AVAILABLE = False

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

from dq_core.models import ProfileMetrics


logger = logging.getLogger(__name__)


@dataclass
class ProfileConfig:
    """Configuration for data profiling."""
    
    compute_statistics: bool = True
    compute_histograms: bool = True
    compute_correlations: bool = False
    sample_size: Optional[int] = None
    profile_numeric: bool = True
    profile_categorical: bool = True
    profile_datetime: bool = True
    max_unique_values: int = 100
    histogram_bins: int = 20


class DataProfiler:
    """Advanced data profiling with multiple backends."""
    
    def __init__(self, config: Optional[ProfileConfig] = None):
        self.config = config or ProfileConfig()
        self.use_whylogs = WHYLOGS_AVAILABLE
        self.use_evidently = EVIDENTLY_AVAILABLE
        
        if not self.use_whylogs and not self.use_evidently:
            logger.warning("Neither WhyLogs nor Evidently available, using basic profiling")
    
    def profile(self, data: Union[pd.DataFrame, Any]) -> List[ProfileMetrics]:
        """Profile the dataset."""
        if isinstance(data, pd.DataFrame):
            return self._profile_pandas(data)
        else:
            # For Spark DataFrames, delegate to adapter
            logger.info("Non-pandas data provided, delegating to adapter")
            return []
    
    def _profile_pandas(self, df: pd.DataFrame) -> List[ProfileMetrics]:
        """Profile pandas DataFrame."""
        profiles = []
        
        # Apply sampling if configured
        if self.config.sample_size and len(df) > self.config.sample_size:
            df_sample = df.sample(n=self.config.sample_size, random_state=42)
        else:
            df_sample = df
        
        # Use WhyLogs if available
        if self.use_whylogs:
            profiles = self._profile_with_whylogs(df_sample)
        # Use Evidently if available
        elif self.use_evidently:
            profiles = self._profile_with_evidently(df_sample)
        # Fallback to basic profiling
        else:
            profiles = self._basic_profile(df_sample)
        
        return profiles
    
    def _profile_with_whylogs(self, df: pd.DataFrame) -> List[ProfileMetrics]:
        """Profile using WhyLogs."""
        profiles = []
        
        # Create WhyLogs profile
        profile_view = why.log(df).view()
        
        for column in df.columns:
            col_profile = profile_view.get_column(column)
            
            if col_profile:
                metrics = ProfileMetrics(
                    column_name=column,
                    data_type=str(df[column].dtype),
                    count=col_profile.get_metric("counts").n.value,
                    null_count=col_profile.get_metric("counts").null.value,
                    null_percentage=(col_profile.get_metric("counts").null.value / len(df) * 100),
                    distinct_count=col_profile.get_metric("cardinality").est.value,
                    distinct_percentage=(col_profile.get_metric("cardinality").est.value / len(df) * 100)
                )
                
                # Numeric metrics
                if pd.api.types.is_numeric_dtype(df[column]):
                    distribution = col_profile.get_metric("distribution")
                    if distribution:
                        metrics.min_value = distribution.min.value
                        metrics.max_value = distribution.max.value
                        metrics.mean = distribution.mean.value
                        metrics.std_dev = distribution.stddev.value
                        metrics.median = distribution.median.value
                        
                        # Quantiles
                        metrics.quantiles = {
                            "q25": distribution.q_25.value,
                            "q50": distribution.median.value,
                            "q75": distribution.q_75.value
                        }
                
                profiles.append(metrics)
        
        return profiles
    
    def _profile_with_evidently(self, df: pd.DataFrame) -> List[ProfileMetrics]:
        """Profile using Evidently."""
        profiles = []
        
        # Create Evidently report
        report = Report(metrics=[DataQualityPreset()])
        report.run(current_data=df, reference_data=None)
        
        # Extract metrics from report
        report_dict = report.as_dict()
        
        for column in df.columns:
            metrics = self._extract_evidently_metrics(df, column, report_dict)
            profiles.append(metrics)
        
        return profiles
    
    def _extract_evidently_metrics(self, df: pd.DataFrame, column: str, 
                                  report_dict: Dict) -> ProfileMetrics:
        """Extract metrics from Evidently report."""
        metrics = ProfileMetrics(
            column_name=column,
            data_type=str(df[column].dtype),
            count=df[column].count(),
            null_count=df[column].isnull().sum(),
            null_percentage=(df[column].isnull().sum() / len(df) * 100),
            distinct_count=df[column].nunique(),
            distinct_percentage=(df[column].nunique() / len(df) * 100)
        )
        
        # Add numeric statistics
        if pd.api.types.is_numeric_dtype(df[column]):
            metrics.min_value = float(df[column].min())
            metrics.max_value = float(df[column].max())
            metrics.mean = float(df[column].mean())
            metrics.median = float(df[column].median())
            metrics.std_dev = float(df[column].std())
            
            quantiles = df[column].quantile([0.25, 0.5, 0.75])
            metrics.quantiles = {
                "q25": float(quantiles[0.25]),
                "q50": float(quantiles[0.5]),
                "q75": float(quantiles[0.75])
            }
        
        return metrics
    
    def _basic_profile(self, df: pd.DataFrame) -> List[ProfileMetrics]:
        """Basic profiling without external libraries."""
        profiles = []
        total_rows = len(df)
        
        for column in df.columns:
            col_data = df[column]
            
            metrics = ProfileMetrics(
                column_name=column,
                data_type=str(col_data.dtype),
                count=col_data.count(),
                null_count=col_data.isnull().sum(),
                null_percentage=(col_data.isnull().sum() / total_rows * 100) if total_rows > 0 else 0,
                distinct_count=col_data.nunique(),
                distinct_percentage=(col_data.nunique() / total_rows * 100) if total_rows > 0 else 0
            )
            
            # Numeric statistics
            if pd.api.types.is_numeric_dtype(col_data) and self.config.profile_numeric:
                non_null = col_data.dropna()
                if len(non_null) > 0:
                    metrics.min_value = float(non_null.min())
                    metrics.max_value = float(non_null.max())
                    metrics.mean = float(non_null.mean())
                    metrics.median = float(non_null.median())
                    metrics.std_dev = float(non_null.std())
                    
                    # Quantiles
                    quantiles = non_null.quantile([0.25, 0.5, 0.75])
                    metrics.quantiles = {
                        "q25": float(quantiles[0.25]),
                        "q50": float(quantiles[0.5]),
                        "q75": float(quantiles[0.75])
                    }
                    
                    # Histogram
                    if self.config.compute_histograms:
                        hist, bins = np.histogram(non_null, bins=self.config.histogram_bins)
                        metrics.histogram = {
                            "counts": hist.tolist(),
                            "bins": bins.tolist()
                        }
            
            # Categorical statistics
            elif pd.api.types.is_object_dtype(col_data) and self.config.profile_categorical:
                # String length statistics
                str_lengths = col_data.astype(str).str.len()
                metrics.min_value = str(str_lengths.min())
                metrics.max_value = str(str_lengths.max())
                
                # Top values
                value_counts = col_data.value_counts()
                top_n = min(10, len(value_counts))
                metrics.top_values = [
                    {"value": str(value), "count": int(count)}
                    for value, count in value_counts.head(top_n).items()
                ]
            
            # Datetime statistics
            elif pd.api.types.is_datetime64_any_dtype(col_data) and self.config.profile_datetime:
                non_null = col_data.dropna()
                if len(non_null) > 0:
                    metrics.min_value = str(non_null.min())
                    metrics.max_value = str(non_null.max())
            
            profiles.append(metrics)
        
        return profiles
    
    def compare_profiles(self, reference: List[ProfileMetrics], 
                        current: List[ProfileMetrics]) -> Dict[str, Any]:
        """Compare two profiles to detect changes."""
        comparison = {
            "timestamp": datetime.utcnow().isoformat(),
            "columns": {}
        }
        
        ref_dict = {p.column_name: p for p in reference}
        curr_dict = {p.column_name: p for p in current}
        
        all_columns = set(ref_dict.keys()) | set(curr_dict.keys())
        
        for column in all_columns:
            col_comparison = {}
            
            if column not in curr_dict:
                col_comparison["status"] = "removed"
            elif column not in ref_dict:
                col_comparison["status"] = "added"
            else:
                ref_prof = ref_dict[column]
                curr_prof = curr_dict[column]
                
                col_comparison["status"] = "modified" if self._has_significant_change(
                    ref_prof, curr_prof
                ) else "unchanged"
                
                # Calculate changes
                if ref_prof.null_percentage != curr_prof.null_percentage:
                    col_comparison["null_change"] = curr_prof.null_percentage - ref_prof.null_percentage
                
                if ref_prof.distinct_count != curr_prof.distinct_count:
                    col_comparison["distinct_change"] = curr_prof.distinct_count - ref_prof.distinct_count
                
                if ref_prof.mean is not None and curr_prof.mean is not None:
                    col_comparison["mean_change"] = curr_prof.mean - ref_prof.mean
                
                if ref_prof.std_dev is not None and curr_prof.std_dev is not None:
                    col_comparison["std_change"] = curr_prof.std_dev - ref_prof.std_dev
            
            comparison["columns"][column] = col_comparison
        
        return comparison
    
    def _has_significant_change(self, ref: ProfileMetrics, 
                               curr: ProfileMetrics,
                               threshold: float = 0.1) -> bool:
        """Check if profile has significant changes."""
        # Check null percentage change
        if abs(curr.null_percentage - ref.null_percentage) > threshold * 100:
            return True
        
        # Check distinct count change
        if ref.distinct_count > 0:
            distinct_change = abs(curr.distinct_count - ref.distinct_count) / ref.distinct_count
            if distinct_change > threshold:
                return True
        
        # Check mean change for numeric columns
        if ref.mean is not None and curr.mean is not None and ref.mean != 0:
            mean_change = abs(curr.mean - ref.mean) / abs(ref.mean)
            if mean_change > threshold:
                return True
        
        # Check std deviation change
        if ref.std_dev is not None and curr.std_dev is not None and ref.std_dev != 0:
            std_change = abs(curr.std_dev - ref.std_dev) / ref.std_dev
            if std_change > threshold:
                return True
        
        return False
    
    def generate_report(self, profiles: List[ProfileMetrics]) -> str:
        """Generate a text report from profiles."""
        report = []
        report.append("=" * 80)
        report.append("DATA PROFILE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.utcnow().isoformat()}")
        report.append(f"Columns: {len(profiles)}")
        report.append("")
        
        for profile in profiles:
            report.append(f"\nColumn: {profile.column_name}")
            report.append("-" * 40)
            report.append(f"  Type: {profile.data_type}")
            report.append(f"  Count: {profile.count:,}")
            report.append(f"  Nulls: {profile.null_count:,} ({profile.null_percentage:.2f}%)")
            report.append(f"  Distinct: {profile.distinct_count:,} ({profile.distinct_percentage:.2f}%)")
            
            if profile.mean is not None:
                report.append(f"  Min: {profile.min_value}")
                report.append(f"  Max: {profile.max_value}")
                report.append(f"  Mean: {profile.mean:.2f}")
                report.append(f"  Median: {profile.median:.2f}")
                report.append(f"  Std Dev: {profile.std_dev:.2f}")
            
            if profile.top_values:
                report.append("  Top Values:")
                for i, item in enumerate(profile.top_values[:5], 1):
                    report.append(f"    {i}. {item['value']}: {item['count']:,}")
        
        return "\n".join(report)