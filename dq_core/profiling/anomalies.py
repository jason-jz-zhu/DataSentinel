"""Statistical and ML-based anomaly detection."""

from typing import Any, Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass

from dq_core.registry import BaseDetector, register_detector
from dq_core.models import AnomalyResult


logger = logging.getLogger(__name__)


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection."""
    
    z_score_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    min_samples: int = 30
    lookback_days: int = 30
    confidence_threshold: float = 0.95
    methods: List[str] = None
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = ["z_score", "iqr", "isolation_forest"]


class AnomalyDetector:
    """Main anomaly detection orchestrator."""
    
    def __init__(self, config: Optional[AnomalyConfig] = None):
        self.config = config or AnomalyConfig()
        self.detectors = self._initialize_detectors()
    
    def _initialize_detectors(self) -> Dict[str, BaseDetector]:
        """Initialize configured detectors."""
        detectors = {}
        
        if "z_score" in self.config.methods:
            detectors["z_score"] = ZScoreDetector(self.config)
        
        if "iqr" in self.config.methods:
            detectors["iqr"] = IQRDetector(self.config)
        
        if "isolation_forest" in self.config.methods:
            try:
                from sklearn.ensemble import IsolationForest
                detectors["isolation_forest"] = IsolationForestDetector(self.config)
            except ImportError:
                logger.warning("scikit-learn not available, skipping IsolationForest")
        
        return detectors
    
    def detect(self, current_metrics: Dict[str, float],
              historical_metrics: pd.DataFrame) -> List[AnomalyResult]:
        """Detect anomalies in current metrics."""
        results = []
        
        for metric_name, current_value in current_metrics.items():
            if metric_name not in historical_metrics.columns:
                continue
            
            historical_values = historical_metrics[metric_name].dropna()
            
            if len(historical_values) < self.config.min_samples:
                logger.warning(f"Insufficient samples for {metric_name}: {len(historical_values)}")
                continue
            
            # Run all configured detectors
            for method_name, detector in self.detectors.items():
                try:
                    result = detector.detect_single(
                        metric_name, current_value, historical_values
                    )
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error in {method_name} detector for {metric_name}: {e}")
        
        return results
    
    def detect_batch(self, metrics_df: pd.DataFrame) -> Dict[str, List[AnomalyResult]]:
        """Detect anomalies in batch of metrics."""
        results = {}
        
        for column in metrics_df.columns:
            if column == "timestamp":
                continue
            
            column_results = []
            values = metrics_df[column].dropna()
            
            if len(values) < self.config.min_samples:
                continue
            
            # Detect anomalies in the entire series
            for method_name, detector in self.detectors.items():
                try:
                    anomalies = detector.detect_series(column, values)
                    column_results.extend(anomalies)
                except Exception as e:
                    logger.error(f"Error in {method_name} detector for {column}: {e}")
            
            if column_results:
                results[column] = column_results
        
        return results


@register_detector("z_score")
class ZScoreDetector(BaseDetector):
    """Z-score based anomaly detection."""
    
    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.mean = None
        self.std = None
    
    def fit(self, data: pd.Series) -> None:
        """Fit the detector on historical data."""
        self.mean = data.mean()
        self.std = data.std()
    
    def detect(self, data: pd.Series) -> List[int]:
        """Detect anomalies in series."""
        if self.mean is None or self.std is None:
            self.fit(data)
        
        z_scores = np.abs((data - self.mean) / self.std)
        anomaly_indices = np.where(z_scores > self.config.z_score_threshold)[0]
        return anomaly_indices.tolist()
    
    def detect_single(self, metric_name: str, current_value: float,
                     historical_values: pd.Series) -> Optional[AnomalyResult]:
        """Detect if single value is anomalous."""
        mean = historical_values.mean()
        std = historical_values.std()
        
        if std == 0:
            return None
        
        z_score = abs((current_value - mean) / std)
        is_anomaly = z_score > self.config.z_score_threshold
        
        # Calculate expected range
        expected_min = mean - (self.config.z_score_threshold * std)
        expected_max = mean + (self.config.z_score_threshold * std)
        
        confidence = min(0.99, 1 - (2 * stats.norm.cdf(-abs(z_score))))
        
        return AnomalyResult(
            metric_name=metric_name,
            current_value=current_value,
            expected_range=(expected_min, expected_max),
            z_score=z_score,
            is_anomaly=is_anomaly,
            confidence=confidence,
            method="z_score",
            details={
                "mean": mean,
                "std": std,
                "threshold": self.config.z_score_threshold
            }
        )
    
    def detect_series(self, metric_name: str, values: pd.Series) -> List[AnomalyResult]:
        """Detect anomalies in entire series."""
        results = []
        mean = values.mean()
        std = values.std()
        
        if std == 0:
            return results
        
        z_scores = np.abs((values - mean) / std)
        anomaly_mask = z_scores > self.config.z_score_threshold
        
        for idx in np.where(anomaly_mask)[0]:
            result = AnomalyResult(
                metric_name=metric_name,
                current_value=float(values.iloc[idx]),
                expected_range=(mean - self.config.z_score_threshold * std,
                              mean + self.config.z_score_threshold * std),
                z_score=float(z_scores.iloc[idx]),
                is_anomaly=True,
                confidence=min(0.99, 1 - (2 * stats.norm.cdf(-abs(z_scores.iloc[idx])))),
                method="z_score",
                details={"index": int(idx)}
            )
            results.append(result)
        
        return results


@register_detector("iqr")
class IQRDetector(BaseDetector):
    """Interquartile Range based anomaly detection."""
    
    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.q1 = None
        self.q3 = None
        self.iqr = None
    
    def fit(self, data: pd.Series) -> None:
        """Fit the detector on historical data."""
        self.q1 = data.quantile(0.25)
        self.q3 = data.quantile(0.75)
        self.iqr = self.q3 - self.q1
    
    def detect(self, data: pd.Series) -> List[int]:
        """Detect anomalies in series."""
        if self.iqr is None:
            self.fit(data)
        
        lower_bound = self.q1 - (self.config.iqr_multiplier * self.iqr)
        upper_bound = self.q3 + (self.config.iqr_multiplier * self.iqr)
        
        anomaly_mask = (data < lower_bound) | (data > upper_bound)
        anomaly_indices = np.where(anomaly_mask)[0]
        return anomaly_indices.tolist()
    
    def detect_single(self, metric_name: str, current_value: float,
                     historical_values: pd.Series) -> Optional[AnomalyResult]:
        """Detect if single value is anomalous."""
        q1 = historical_values.quantile(0.25)
        q3 = historical_values.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:
            return None
        
        lower_bound = q1 - (self.config.iqr_multiplier * iqr)
        upper_bound = q3 + (self.config.iqr_multiplier * iqr)
        
        is_anomaly = current_value < lower_bound or current_value > upper_bound
        
        # Estimate confidence based on distance from bounds
        if is_anomaly:
            if current_value < lower_bound:
                distance = (lower_bound - current_value) / iqr
            else:
                distance = (current_value - upper_bound) / iqr
            confidence = min(0.99, 0.5 + (distance * 0.1))
        else:
            confidence = 0.0
        
        return AnomalyResult(
            metric_name=metric_name,
            current_value=current_value,
            expected_range=(lower_bound, upper_bound),
            is_anomaly=is_anomaly,
            confidence=confidence,
            method="iqr",
            details={
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "multiplier": self.config.iqr_multiplier
            }
        )
    
    def detect_series(self, metric_name: str, values: pd.Series) -> List[AnomalyResult]:
        """Detect anomalies in entire series."""
        results = []
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:
            return results
        
        lower_bound = q1 - (self.config.iqr_multiplier * iqr)
        upper_bound = q3 + (self.config.iqr_multiplier * iqr)
        
        anomaly_mask = (values < lower_bound) | (values > upper_bound)
        
        for idx in np.where(anomaly_mask)[0]:
            value = float(values.iloc[idx])
            if value < lower_bound:
                distance = (lower_bound - value) / iqr
            else:
                distance = (value - upper_bound) / iqr
            
            result = AnomalyResult(
                metric_name=metric_name,
                current_value=value,
                expected_range=(lower_bound, upper_bound),
                is_anomaly=True,
                confidence=min(0.99, 0.5 + (distance * 0.1)),
                method="iqr",
                details={"index": int(idx)}
            )
            results.append(result)
        
        return results


@register_detector("isolation_forest")
class IsolationForestDetector(BaseDetector):
    """Isolation Forest based anomaly detection."""
    
    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.model = None
        
        try:
            from sklearn.ensemble import IsolationForest
            self.IsolationForest = IsolationForest
        except ImportError:
            raise ImportError("scikit-learn required for IsolationForest detector")
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the detector on historical data."""
        self.model = self.IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.model.fit(data.values.reshape(-1, 1))
    
    def detect(self, data: pd.Series) -> List[int]:
        """Detect anomalies in series."""
        if self.model is None:
            self.fit(pd.DataFrame(data))
        
        predictions = self.model.predict(data.values.reshape(-1, 1))
        anomaly_indices = np.where(predictions == -1)[0]
        return anomaly_indices.tolist()
    
    def detect_single(self, metric_name: str, current_value: float,
                     historical_values: pd.Series) -> Optional[AnomalyResult]:
        """Detect if single value is anomalous."""
        # Train model on historical data
        model = self.IsolationForest(contamination=0.1, random_state=42)
        X = historical_values.values.reshape(-1, 1)
        model.fit(X)
        
        # Predict on current value
        prediction = model.predict([[current_value]])[0]
        score = model.score_samples([[current_value]])[0]
        
        is_anomaly = prediction == -1
        
        # Estimate expected range from historical data
        lower_bound = historical_values.quantile(0.05)
        upper_bound = historical_values.quantile(0.95)
        
        # Convert anomaly score to confidence
        confidence = min(0.99, abs(score))
        
        return AnomalyResult(
            metric_name=metric_name,
            current_value=current_value,
            expected_range=(lower_bound, upper_bound),
            is_anomaly=is_anomaly,
            confidence=confidence,
            method="isolation_forest",
            details={
                "anomaly_score": float(score),
                "prediction": int(prediction)
            }
        )
    
    def detect_series(self, metric_name: str, values: pd.Series) -> List[AnomalyResult]:
        """Detect anomalies in entire series."""
        results = []
        
        # Train model
        model = self.IsolationForest(contamination=0.1, random_state=42)
        X = values.values.reshape(-1, 1)
        model.fit(X)
        
        # Predict anomalies
        predictions = model.predict(X)
        scores = model.score_samples(X)
        
        # Get expected range
        lower_bound = values.quantile(0.05)
        upper_bound = values.quantile(0.95)
        
        anomaly_indices = np.where(predictions == -1)[0]
        
        for idx in anomaly_indices:
            result = AnomalyResult(
                metric_name=metric_name,
                current_value=float(values.iloc[idx]),
                expected_range=(lower_bound, upper_bound),
                is_anomaly=True,
                confidence=min(0.99, abs(scores[idx])),
                method="isolation_forest",
                details={
                    "index": int(idx),
                    "anomaly_score": float(scores[idx])
                }
            )
            results.append(result)
        
        return results