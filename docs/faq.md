# FAQ

### What model types does ModelDoctor support?
ModelDoctor officially supports **scikit-learn** compatible estimators (`BaseEstimator`). This includes models from Scikit-Learn, XGBoost, LightGBM, and CatBoost (when using their sklearn wrappers).

### Does it support regression?
Yes. ModelDoctor automatically infers whether the task is classification or regression and routes the context to the appropriate diagnostic logic within the Doctors.

### Why are SHAP charts missing from my dashboard?
The `ExplainabilityEngine` requires the `shap` library. If it is not installed, ModelDoctor gracefully falls back to scikit-learn's native permutation importance and skips SHAP rendering. Install it via `pip install "modeldoctor[shap]"`.

### The dashboard won't open when I call `report.dashboard()`
Some operating systems or notebook environments restrict the opening of temporary HTML files. You can bypass this by manually saving the file: `report.save_html("report.html")` and opening it in your browser.

### Is ModelDoctor slow on large datasets?
ModelDoctor lazily computes expensive metrics (like cross-validation or permutation importance). However, if a Doctor requests these metrics on a dataset with millions of rows, execution can take several minutes. We recommend passing a stratified sample (e.g., 50k rows) to `md.diagnose()` for rapid iteration.

### Can I compare two models side-by-side?
ModelDoctor evaluates a single model pipeline per `diagnose()` call. To compare models, generate two reports and view the HTML dashboards in separate browser tabs.

### Are you planning to support PyTorch/TensorFlow?
Yes, deep learning support is on the roadmap. Neural networks present unique diagnostic challenges (vanishing gradients, catastrophic forgetting) which we are currently building dedicated Doctors for.
