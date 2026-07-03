# Installation

ModelDoctor requires **Python 3.9+**. 

## Standard Installation

To install the core diagnostic engine:

```bash
pip install modeldoctor
```

## Optional Extras

ModelDoctor provides optional dependencies for extended functionality:

```bash
# Install dashboard dependencies (Jinja2, HTML rendering)
pip install "modeldoctor[dashboard]"

# Install explainability dependencies (SHAP)
pip install "modeldoctor[shap]"

# Install all optional dependencies
pip install "modeldoctor[all]"
```

## Development Installation

To install ModelDoctor from source for development or contribution:

```bash
git clone https://github.com/modeldoctor/modeldoctor.git
cd modeldoctor

# Install in editable mode with dev dependencies (pytest, black, mkdocs)
pip install -e ".[dev]"
```

## Dependency Notes

ModelDoctor relies on the following core dependencies:
- `scikit-learn` >= 1.0
- `numpy` >= 1.20
- `pandas` >= 1.3
- `pydantic` >= 2.0

### SHAP Support
If you want ModelDoctor to automatically compute and render SHAP feature importance charts, you must install the `[shap]` extra. Without it, ModelDoctor will fallback to native permutation importance.

### Dashboard Support
The HTML dashboard relies on `Jinja2` for rendering templates. If you plan to call `report.dashboard()` or export HTML reports, ensure the `[dashboard]` extra is installed.

---

## Common Installation Errors

### `ModuleNotFoundError: No module named 'shap'`
You are trying to run the `ExplainabilityEngine` without SHAP installed. Run `pip install "modeldoctor[shap]"`.

### `TemplateNotFound` Error
You are attempting to render the HTML dashboard but the Jinja2 templates are missing or the library was installed without dashboard support. Run `pip install "modeldoctor[dashboard]"`.

### Pydantic Validation Errors on Import
ModelDoctor uses Pydantic v2. If you have Pydantic v1 installed, you may see compatibility errors. Upgrade using `pip install -U pydantic`.
