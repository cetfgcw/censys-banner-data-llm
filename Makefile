.PHONY: help build up down logs test evaluate benchmark clean

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

build: ## Build Docker image
	docker-compose build

up: ## Start the service
	docker-compose up -d

down: ## Stop the service
	docker-compose down

logs: ## View service logs
	docker-compose logs -f banner-classifier

test: ## Run tests
	pytest tests/ -v

test-api: ## Test API endpoints
	python scripts/test_api.py

evaluate: ## Run evaluation on dataset
	python scripts/evaluate.py --data banner_data_train.csv --output evaluation_results.json

benchmark: ## Run performance benchmarks
	python scripts/benchmark.py --data banner_data_train.csv --samples 100

clean: ## Clean up Docker resources
	docker-compose down -v
	docker system prune -f

