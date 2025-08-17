# DataSentinel 🛡️

**Enterprise-grade Data Quality Framework for Spark, Snowflake, and S3**

DataSentinel is a production-ready, extensible data quality framework designed for enterprise environments. It combines rule-based validation with ML-powered anomaly detection, providing comprehensive quality monitoring across your data ecosystem.

## ✨ Features

- **Multi-Source Support**: Spark DataFrames, Snowflake warehouses, and S3 data lakes
- **Rule-Based Checks**: Comprehensive library of data quality expectations
- **Anomaly Detection**: Statistical and ML-based profiling with drift detection
- **DQ Scoring**: Multi-dimensional scoring (Accuracy, Reliability, Stewardship, Usability)
- **Enterprise Ready**: Secure by default, auditable, config-driven, and pluggable
- **CI/CD Integration**: Built-in support for data quality gates
- **Real-time Dashboard**: Web UI for monitoring and analysis
- **Extensible**: Plugin architecture for custom checks and detectors

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DataSentinel CLI/API                         │
├─────────────────────────────────────────────────────────────────┤
│  DQ Scorer  │  Rule Engine  │  Profiler  │  Anomaly Detector  │
├─────────────────────────────────────────────────────────────────┤
│                      Adapter Layer                              │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐                 │
│  │  Spark   │  │  Snowflake   │  │   S3     │                 │
│  └──────────┘  └──────────────┘  └──────────┘                 │
├─────────────────────────────────────────────────────────────────┤
│  History Store │  Config Mgmt │  Lineage  │  Alerting        │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Poetry for dependency management
- Docker & Docker Compose (for local development)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/DataSentinel.git
   cd DataSentinel
   ```

2. **Install dependencies**
   ```bash
   poetry install
   ```

3. **Set up pre-commit hooks**
   ```bash
   poetry run pre-commit install
   ```

### Run the Demo

```bash
# Start local services (MinIO, PostgreSQL)
make docker-up

# Run demo with sample data
make run-demo

# Start API and dashboard
make serve
```

Visit:
- **API Documentation**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501

## 📊 Data Quality Dimensions

DataSentinel evaluates data quality across four key dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **Accuracy** | 40% | Data correctness, validity, and constraint compliance |
| **Reliability** | 30% | Data freshness, timeliness, and SLA adherence |
| **Stewardship** | 15% | Data ownership, documentation, and governance |
| **Usability** | 15% | Schema consistency, accessibility, and ease of use |

## 🔧 Configuration

### Dataset Configuration (`demo_local.yaml`)

```yaml
settings:
  environment: local
  database:
    host: localhost
    port: 5432
    database: dqmetrics
  
datasets:
  - name: customers
    storage_type: spark
    location: s3://dq-data/customers/
    owner: data-team@company.com
    sla_hours: 24
    checks:
      - not_null_customer_id
      - unique_email
      - valid_phone_format
    profiling_enabled: true
    anomaly_detection_enabled: true
```

## 🛡️ Built-in Checks

### Data Validation
- `not_null`: Check for null values
- `unique`: Verify uniqueness constraints
- `range`: Validate numeric ranges
- `values_in_set`: Enum validation
- `regex`: Pattern matching
- `foreign_key`: Referential integrity

### Data Freshness
- `freshness`: Data recency checks
- `landing_time`: SLA compliance
- `update_frequency`: Regular update validation

### Schema Validation
- `schema_match`: Schema compliance
- `schema_drift`: Change detection
- `column_types`: Type validation

## 🎯 Usage Examples

### Command Line Interface

```bash
# Run data quality scan
dq scan --config config.yaml --dataset customers

# Get DQ score
dq score --dataset customers --format json

# Start services
dq serve --host 0.0.0.0 --port 8000
```

### Python API

```python
from dq_core import DataSentinel, Config

# Initialize framework
config = Config.from_yaml("config.yaml")
dq = DataSentinel(config)

# Run quality check
result = dq.scan_dataset("customers")
print(f"DQ Score: {result.score.overall_score:.2f}")

# Get historical trends
trends = dq.get_trends("customers", days=30)
```

### CI/CD Integration

```yaml
# GitHub Actions
- name: Data Quality Gate
  run: |
    dq scan --config .dq/config.yaml --dataset users
    score=$(dq score --dataset users --format json | jq -r '.overall_score')
    if (( $(echo "$score < 0.8" | bc -l) )); then
      echo "DQ Score below threshold"
      exit 1
    fi
```

## 📈 Monitoring & Alerting

### Dashboard Features
- **Real-time Scores**: Live DQ score monitoring
- **Trend Analysis**: Historical score trends
- **Check Status**: Detailed validation results
- **Anomaly Detection**: Automatic outlier identification
- **Alert Management**: Issue tracking and resolution

### Alerting Channels
- Slack webhooks
- PagerDuty integration
- Email notifications
- Custom webhooks

## 🔌 Extensibility

### Custom Checks

```python
from dq_core.registry import register_check, BaseCheck

@register_check("custom_business_rule")
class CustomBusinessCheck(BaseCheck):
    def run(self, data, **params):
        # Implement custom logic
        return {"passed": True, "details": "Custom check passed"}
```

### Custom Detectors

```python
from dq_core.registry import register_detector, BaseDetector

@register_detector("ml_anomaly")
class MLAnomalyDetector(BaseDetector):
    def detect(self, data):
        # Implement ML-based detection
        return anomaly_results
```

## 🔒 Security & Compliance

- **PII Protection**: Automatic data redaction and sampling
- **Audit Logging**: Comprehensive activity tracking
- **Access Control**: Role-based permissions
- **Encryption**: Data at rest and in transit
- **SOC 2 Ready**: Enterprise security standards

## 📦 Deployment

### Docker

```bash
# Build image
docker build -t datasentinel .

# Run container
docker run -p 8000:8000 -p 8501:8501 datasentinel
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: datasentinel
spec:
  replicas: 3
  selector:
    matchLabels:
      app: datasentinel
  template:
    metadata:
      labels:
        app: datasentinel
    spec:
      containers:
      - name: datasentinel
        image: datasentinel:latest
        ports:
        - containerPort: 8000
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
poetry run pytest --cov=dq_core --cov-report=html

# Integration tests
make test-integration
```

## 📚 Documentation

- [Configuration Guide](docs/configuration.md)
- [Check Library](docs/checks.md)
- [API Reference](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [Contributing](docs/contributing.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙋‍♀️ Support

- **Documentation**: [docs.datasentinel.io](https://docs.datasentinel.io)
- **Issues**: [GitHub Issues](https://github.com/your-org/DataSentinel/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/DataSentinel/discussions)
- **Slack**: [DataSentinel Community](https://datasentinel.slack.com)

---

**Built with ❤️ for the data community**