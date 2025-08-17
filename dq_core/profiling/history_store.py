"""Historical metrics storage for trend analysis."""

from typing import Any, Dict, List, Optional
import logging
from datetime import datetime, timedelta
import json

import pandas as pd
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from dq_core.config import DatabaseConfig


logger = logging.getLogger(__name__)
Base = declarative_base()


class MetricRecord(Base):
    """SQLAlchemy model for metric history."""
    
    __tablename__ = "metrics_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_name = Column(String(255), nullable=False, index=True)
    metric_name = Column(String(255), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    metadata = Column(Text, nullable=True)


class MetricsHistoryStore:
    """Store and retrieve historical metrics."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.engine = None
        self.session_factory = None
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Initialize database connection."""
        db_url = (
            f"postgresql://{self.config.username}:{self.config.password}@"
            f"{self.config.host}:{self.config.port}/{self.config.database}"
        )
        
        self.engine = create_engine(
            db_url,
            pool_size=self.config.pool_size,
            echo=self.config.echo
        )
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
        self.session_factory = sessionmaker(bind=self.engine)
        logger.info("Metrics history store initialized")
    
    def store_metrics(self, dataset_name: str, metrics: Dict[str, float],
                     timestamp: Optional[datetime] = None) -> None:
        """Store metrics for a dataset."""
        timestamp = timestamp or datetime.utcnow()
        session = self.session_factory()
        
        try:
            for metric_name, metric_value in metrics.items():
                record = MetricRecord(
                    dataset_name=dataset_name,
                    metric_name=metric_name,
                    metric_value=float(metric_value),
                    timestamp=timestamp
                )
                session.add(record)
            
            session.commit()
            logger.debug(f"Stored {len(metrics)} metrics for {dataset_name}")
        
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to store metrics: {e}")
            raise
        
        finally:
            session.close()
    
    def get_metrics(self, dataset_name: str, metric_name: Optional[str] = None,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None,
                   limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve historical metrics."""
        session = self.session_factory()
        
        try:
            query = session.query(MetricRecord).filter(
                MetricRecord.dataset_name == dataset_name
            )
            
            if metric_name:
                query = query.filter(MetricRecord.metric_name == metric_name)
            
            if start_date:
                query = query.filter(MetricRecord.timestamp >= start_date)
            
            if end_date:
                query = query.filter(MetricRecord.timestamp <= end_date)
            
            query = query.order_by(MetricRecord.timestamp.desc())
            
            if limit:
                query = query.limit(limit)
            
            records = query.all()
            
            # Convert to DataFrame
            data = []
            for record in records:
                data.append({
                    "timestamp": record.timestamp,
                    "metric_name": record.metric_name,
                    "metric_value": record.metric_value
                })
            
            df = pd.DataFrame(data)
            
            # Pivot to have metrics as columns
            if not df.empty and metric_name is None:
                df = df.pivot(
                    index="timestamp",
                    columns="metric_name",
                    values="metric_value"
                ).reset_index()
            
            return df
        
        finally:
            session.close()
    
    def get_latest_metrics(self, dataset_name: str) -> Dict[str, float]:
        """Get the latest metrics for a dataset."""
        session = self.session_factory()
        
        try:
            # Get the latest timestamp
            latest_timestamp = session.query(
                MetricRecord.timestamp
            ).filter(
                MetricRecord.dataset_name == dataset_name
            ).order_by(
                MetricRecord.timestamp.desc()
            ).first()
            
            if not latest_timestamp:
                return {}
            
            # Get all metrics for that timestamp
            records = session.query(MetricRecord).filter(
                MetricRecord.dataset_name == dataset_name,
                MetricRecord.timestamp == latest_timestamp[0]
            ).all()
            
            return {
                record.metric_name: record.metric_value
                for record in records
            }
        
        finally:
            session.close()
    
    def cleanup_old_metrics(self, retention_days: int = 90) -> int:
        """Remove metrics older than retention period."""
        session = self.session_factory()
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            deleted = session.query(MetricRecord).filter(
                MetricRecord.timestamp < cutoff_date
            ).delete()
            
            session.commit()
            logger.info(f"Deleted {deleted} old metric records")
            return deleted
        
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to cleanup metrics: {e}")
            raise
        
        finally:
            session.close()
    
    def get_metric_statistics(self, dataset_name: str, metric_name: str,
                             days: int = 30) -> Dict[str, float]:
        """Get statistics for a metric over time."""
        start_date = datetime.utcnow() - timedelta(days=days)
        df = self.get_metrics(
            dataset_name=dataset_name,
            metric_name=metric_name,
            start_date=start_date
        )
        
        if df.empty:
            return {}
        
        values = df["metric_value"] if "metric_value" in df.columns else df[metric_name]
        
        return {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "count": len(values)
        }