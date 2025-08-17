.PHONY: help install dev-install lint test clean run-demo serve docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  install       Install production dependencies"
	@echo "  dev-install   Install development dependencies"
	@echo "  lint          Run linting and type checking"
	@echo "  test          Run tests"
	@echo "  clean         Clean up generated files"
	@echo "  run-demo      Run demo with local data"
	@echo "  serve         Start API and dashboard"
	@echo "  docker-up     Start local services (MinIO, PostgreSQL)"
	@echo "  docker-down   Stop local services"

install:
	poetry install --only main

dev-install:
	poetry install
	pre-commit install

lint:
	poetry run ruff check dq_core tests
	poetry run mypy dq_core
	poetry run black --check dq_core tests

format:
	poetry run ruff check --fix dq_core tests
	poetry run black dq_core tests

test:
	poetry run pytest tests/ -v --cov=dq_core --cov-report=term-missing

test-integration:
	docker-compose up -d
	sleep 5
	poetry run pytest tests/ -v -m integration
	docker-compose down

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov .pytest_cache
	rm -rf .mypy_cache .ruff_cache
	rm -rf dq_results/ metrics_history.db

run-demo: docker-up
	@echo "Setting up demo data..."
	poetry run python -m dq_core.cli.setup_demo
	@echo "Running DQ scan on demo data..."
	poetry run dq scan --config examples/configs/demo_local.yaml --dataset customers
	poetry run dq scan --config examples/configs/demo_local.yaml --dataset transactions
	@echo "Computing DQ scores..."
	poetry run dq score --dataset customers
	poetry run dq score --dataset transactions
	@echo "Demo complete! Run 'make serve' to view dashboard"

serve:
	@echo "Starting DataSentinel services..."
	@trap 'kill %1; kill %2' INT; \
	poetry run uvicorn dq_core.api.server:app --reload --port 8000 & \
	poetry run streamlit run dq_core/dashboard/app.py --server.port 8501 & \
	wait

docker-up:
	docker-compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 5
	@echo "Services started: MinIO (9000), PostgreSQL (5432)"

docker-down:
	docker-compose down -v

init-minio: docker-up
	@echo "Creating MinIO buckets..."
	docker exec -it datasentinel-minio mc alias set local http://localhost:9000 minioadmin minioadmin
	docker exec -it datasentinel-minio mc mb local/dq-data || true
	docker exec -it datasentinel-minio mc mb local/dq-results || true
	@echo "MinIO buckets created"

spark-shell:
	poetry run pyspark --conf spark.sql.adaptive.enabled=true \
		--conf spark.sql.adaptive.coalescePartitions.enabled=true

notebook:
	poetry run jupyter notebook examples/notebooks/

pre-commit:
	pre-commit run --all-files

version:
	@poetry version

build:
	poetry build

publish-test:
	poetry publish -r test-pypi

publish:
	poetry publish