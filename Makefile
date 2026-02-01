.PHONY: install lint format fix check clean demo

# Install dependencies
install:
	uv pip install -r requirements.txt

install-dev:
	uv pip install -e ".[dev]"

# Linting
lint:
	ruff check .

# Formatting (check only)
format-check:
	ruff format --check .

# Auto-fix all issues (lint + format)
fix:
	ruff check --fix .
	ruff format .

# Alias for fix
format: fix

# Run all checks (lint + format check + type check)
check:
	ruff check .
	ruff format --check .
	mypy grav_font/ --ignore-missing-imports

# Clean cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Demo commands
demo:
	python run_demo.py --demo-no-font

demo-animate:
	python run_demo.py --demo-no-font --animate

demo-all:
	python run_demo.py --demo-all-presets
