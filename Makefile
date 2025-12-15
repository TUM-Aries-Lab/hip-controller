SHELL := /bin/bash

init:  # ENV SETUP
	uv sync --all-groups
	uv run pre-commit install
	@echo "Environment initialized with uv."

test:
	uv run pytest --cov=src --cov-report=term-missing --no-cov-on-fail --cov-report=xml --cov-fail-under=90
	rm .coverage

lint:
	uv run ruff format
	uv run ruff check --fix

typecheck:
	uv run pyright src

format:
	make lint
	make typecheck

clean:
	rm -rf .venv
	rm -rf .pytest_cache
	rm -rf build/
	rm -rf dist/
	rm -rf junit-pytest.xml
	rm -rf logs/*
	find . -name ".coverage*" -delete
	find . -name "coverage.xml" -delete
	find . -name "__pycache__" -exec rm -r {} +

update:
	uv sync --upgrade --all-groups
	uv run pre-commit autoupdate

update-deep:
	uv cache clean pypi
	make update

docker:
	docker build --no-cache -f Dockerfile -t hip_controller-smoke .
	docker run --rm hip_controller-smoke

app:
	uv run python -m hip_controller

tree:
	uv run python repo_tree.py --update-readme

build:
	uv build
	unzip -l dist/*.whl
	unzip -p dist/*.whl */METADATA

