"""Setup demo data and environment."""

import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)


def setup_demo():
    """Setup demo environment."""
    print("🛡️ Setting up DataSentinel Demo Environment")
    
    # Create necessary directories
    dirs = [
        "dq_results",
        "logs",
        "schemas"
    ]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}")
    
    # Set environment variables for demo
    os.environ.setdefault("DQ_ENVIRONMENT", "local")
    os.environ.setdefault("DQ_DEBUG", "true")
    
    print("✅ Demo environment ready!")
    print("📊 Sample datasets available:")
    print("   - customers.csv (10 records)")
    print("   - transactions.csv (10 records)")
    print("\n🚀 Run the following commands:")
    print("   dq scan --config examples/configs/demo_local.yaml --dataset customers")
    print("   dq score --dataset customers")
    print("   dq serve")


if __name__ == "__main__":
    setup_demo()