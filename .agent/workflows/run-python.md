---
description: How to run Python scripts and commands in this project
---

# Running Python in this project

**ALWAYS use the project's virtual environment** located at `.venv` within the project root.

## Python executable

Use this path to run any Python script or command:

```
.\.venv\Scripts\python.exe
```

## Examples

// turbo-all

1. Run a script:
```
.\.venv\Scripts\python.exe scripts\some_script.py
```

2. Run a long-running script with **real-time output** (unbuffered):
```
.\.venv\Scripts\python.exe -u scripts\some_script.py
```

2. Run a one-liner:
```
.\.venv\Scripts\python.exe -c "print('hello')"
```

3. Install a package:
```
.\.venv\Scripts\pip.exe install some-package
```

## Why

- The `.venv` has CUDA-enabled PyTorch (NVIDIA GeForce RTX 4070 Laptop GPU)
- The system Python does NOT have CUDA support
- Using the wrong Python causes training to run on CPU (~8x slower) and may produce different numerical results due to different PyTorch builds
