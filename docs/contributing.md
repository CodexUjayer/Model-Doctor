# Contributing

We welcome contributions to ModelDoctor! Whether you are adding a new Doctor, fixing a bug, or improving the documentation, your help is appreciated.

## Development Setup

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/modeldoctor.git
   cd modeldoctor
   ```
3. **Install in editable mode** with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Running Tests

ModelDoctor uses `pytest`. Ensure all tests pass before submitting a Pull Request:
```bash
pytest tests/
```

## Running the Validation Laboratory

If you modify the core diagnostic engine or threshold rules, you **must** run the Validation Laboratory to ensure you haven't caused regressions in diagnostic accuracy:
```bash
python validation/benchmark_runner.py
```
Your PR must maintain or improve the existing Validation Accuracy (currently 98.1%).

## Coding Standards

- We follow **PEP 8**. Use `black` for formatting.
- All public functions and classes must have docstrings.
- Ensure type hints are used throughout the codebase.

## Submitting a PR

1. Create a feature branch: `git checkout -b feature/my-new-doctor`.
2. Commit your changes with clear messages.
3. Push to your fork and open a Pull Request against the `main` branch.
4. Ensure the CI pipeline (GitHub Actions) passes.

Thank you for helping improve ModelDoctor!
