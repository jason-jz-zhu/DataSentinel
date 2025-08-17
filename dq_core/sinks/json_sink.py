"""JSON file sink for results."""

import json
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

from dq_core.registry import BaseSink, register_sink


@register_sink("json")
class JSONSink(BaseSink):
    """Write results to JSON files."""
    
    def __init__(self, output_path: str = "./dq_results"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
    
    def write(self, result: Any) -> None:
        """Write result to JSON file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"dq_result_{timestamp}.json"
        filepath = self.output_path / filename
        
        with open(filepath, "w") as f:
            json.dump(result.model_dump() if hasattr(result, "model_dump") else result, 
                     f, indent=2, default=str)
    
    def read(self, query: Dict[str, Any]) -> List[Any]:
        """Read results from JSON files."""
        results = []
        
        for file_path in self.output_path.glob("*.json"):
            with open(file_path) as f:
                data = json.load(f)
                results.append(data)
        
        return results