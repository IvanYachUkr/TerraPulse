# Coding Style Guide

## Python Formatting

- **Formatter**: [Black](https://github.com/psf/black) (line length 100)
- **Import sorting**: [isort](https://pycqa.github.io/isort/) (Black-compatible profile)
- **Linter**: [flake8](https://flake8.pycqa.org/) (line length 100)

### Setup

```bash
pip install black isort flake8
```

### Usage

```bash
# Format all Python files
black --line-length 100 .
isort --profile black .

# Lint check
flake8 --max-line-length 100 .
```

## Code Conventions

- **Docstrings**: Every function and module gets a docstring (Google style)
- **Type hints**: Use on all function signatures
- **Naming**:
  - `snake_case` for functions, variables, files
  - `PascalCase` for classes
  - `UPPER_SNAKE_CASE` for constants
- **Notebooks vs scripts**: Scripts for pipeline code (`src/`), notebooks only for exploration (`notebooks/`)
- **No magic numbers**: Constants go at the top of the file or in a config
- **Random seeds**: Always set and document (`RANDOM_SEED = 42`)

## Commit Message Style

```
ACTION: Short description (max 72 chars)

Optional longer explanation if needed.
```

### Action Prefixes

| Prefix | When to Use |
|--------|------------|
| `ADD:` | New file, feature, or functionality |
| `FIX:` | Bug fix |
| `UPDATE:` | Modify existing code or docs |
| `REMOVE:` | Delete files or functionality |
| `REFACTOR:` | Code restructuring, no behavior change |
| `DOCS:` | Documentation only |
| `TEST:` | Adding or updating tests |
| `DATA:` | Data pipeline or download changes |
| `STYLE:` | Formatting, no code change |

### Examples

```
ADD: feature engineering pipeline for spectral indices
FIX: NDVI calculation using wrong band for NIR
UPDATE: grid resolution from 200m to 100m
REMOVE: deprecated CORINE download script
REFACTOR: split data preprocessing into separate modules
DOCS: add evaluation metrics to report
DATA: add Sentinel-2 2023-2025 composites
```

## Branch Strategy

- `main` — stable, release-ready code
- Feature branches: `feature/descriptive-name` (e.g., `feature/catboost-model`)
- Bug fixes: `fix/descriptive-name`
- All merges via Pull Request with at least 1 review
