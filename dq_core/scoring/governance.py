"""Governance signals for DQ scoring."""

from typing import Dict, List, Optional
from datetime import datetime

from dq_core.models import Signal, DQDimension


class GovernanceSignals:
    """Collect and manage governance signals for DQ scoring."""
    
    def __init__(self):
        self.signals: List[Signal] = []
    
    def collect_ownership_signals(self, dataset_config: Dict) -> List[Signal]:
        """Collect ownership-related signals."""
        signals = []
        
        # Data owner presence
        if dataset_config.get("owner"):
            signals.append(Signal(
                name="has_owner",
                value=True,
                dimension=DQDimension.STEWARDSHIP,
                weight=0.3,
                source="config"
            ))
        else:
            signals.append(Signal(
                name="has_owner",
                value=False,
                dimension=DQDimension.STEWARDSHIP,
                weight=0.3,
                source="config"
            ))
        
        # SLA definition
        if dataset_config.get("sla_hours"):
            signals.append(Signal(
                name="has_sla",
                value=True,
                dimension=DQDimension.RELIABILITY,
                weight=0.2,
                source="config"
            ))
        
        # Documentation presence
        if dataset_config.get("description"):
            signals.append(Signal(
                name="has_description",
                value=True,
                dimension=DQDimension.USABILITY,
                weight=0.2,
                source="config"
            ))
        
        return signals
    
    def collect_metadata_signals(self, dataset_config: Dict) -> List[Signal]:
        """Collect metadata-related signals."""
        signals = []
        
        # Tags presence
        tags = dataset_config.get("tags", [])
        if tags:
            signals.append(Signal(
                name="has_tags",
                value=True,
                dimension=DQDimension.USABILITY,
                weight=0.1,
                source="config"
            ))
            
            # PII tag awareness
            if any("pii" in tag.lower() for tag in tags):
                signals.append(Signal(
                    name="pii_tagged",
                    value=True,
                    dimension=DQDimension.STEWARDSHIP,
                    weight=0.2,
                    source="config"
                ))
        
        return signals