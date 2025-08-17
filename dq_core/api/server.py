"""FastAPI server for DataSentinel API."""

from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dq_core import __version__


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


class DatasetInfo(BaseModel):
    name: str
    storage_type: str
    location: str
    owner: Optional[str] = None
    last_scan: Optional[str] = None
    score: Optional[float] = None


class ScanRequest(BaseModel):
    dataset: str
    config_path: str
    checks: Optional[List[str]] = None


app = FastAPI(
    title="DataSentinel API",
    description="Enterprise Data Quality Framework API",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/datasets", response_model=List[DatasetInfo])
async def list_datasets():
    """List all configured datasets."""
    # Mock data for demo
    return [
        DatasetInfo(
            name="customers",
            storage_type="local",
            location="examples/data/customers.csv",
            owner="data-team@company.com",
            last_scan="2023-11-15T10:30:00Z",
            score=0.85
        ),
        DatasetInfo(
            name="transactions",
            storage_type="local",
            location="examples/data/transactions.csv",
            owner="data-team@company.com",
            last_scan="2023-11-15T09:15:00Z",
            score=0.92
        )
    ]


@app.get("/datasets/{dataset_name}")
async def get_dataset(dataset_name: str):
    """Get dataset details."""
    if dataset_name not in ["customers", "transactions"]:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Mock data
    return {
        "name": dataset_name,
        "storage_type": "local",
        "owner": "data-team@company.com",
        "last_scan": "2023-11-15T10:30:00Z",
        "score": 0.85,
        "checks": [
            {"name": "not_null_id", "status": "passed", "pass_rate": 1.0},
            {"name": "unique_id", "status": "passed", "pass_rate": 1.0},
            {"name": "valid_email", "status": "failed", "pass_rate": 0.8}
        ]
    }


@app.post("/scan")
async def run_scan(request: ScanRequest):
    """Run a data quality scan."""
    # Mock response
    return {
        "scan_id": "scan_123",
        "dataset": request.dataset,
        "status": "completed",
        "started_at": "2023-11-15T10:30:00Z",
        "completed_at": "2023-11-15T10:35:00Z",
        "results": {
            "total_checks": 5,
            "passed_checks": 4,
            "failed_checks": 1,
            "score": 0.85
        }
    }


@app.get("/datasets/{dataset_name}/score")
async def get_score(dataset_name: str):
    """Get DQ score for a dataset."""
    if dataset_name not in ["customers", "transactions"]:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Mock score
    return {
        "dataset_name": dataset_name,
        "overall_score": 0.85,
        "grade": "B+",
        "dimensions": {
            "accuracy": 0.90,
            "reliability": 0.80,
            "stewardship": 0.75,
            "usability": 0.85
        },
        "passed_checks": 15,
        "failed_checks": 3,
        "total_checks": 18,
        "timestamp": "2023-11-15T10:30:00Z"
    }


@app.get("/datasets/{dataset_name}/profile")
async def get_profile(dataset_name: str):
    """Get dataset profile."""
    if dataset_name not in ["customers", "transactions"]:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Mock profile
    return {
        "dataset_name": dataset_name,
        "row_count": 1000,
        "column_count": 8,
        "columns": [
            {
                "name": "customer_id",
                "type": "integer",
                "null_percentage": 0.0,
                "distinct_count": 1000
            },
            {
                "name": "email",
                "type": "string",
                "null_percentage": 2.5,
                "distinct_count": 975
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)