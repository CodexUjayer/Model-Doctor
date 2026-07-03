# Report API

## `Report`

The `Report` object encapsulates the final diagnostic results and provides export utilities.

### Properties

- **`health_score`** (`float`): The overall 0-100 diagnostic score.
- **`findings`** (`List[Finding]`): The raw diagnostic evidence.
- **`prescriptions`** (`List[Prescription]`): Generated actionable recommendations.
- **`model_passport`** (`ModelPassport`): Metadata about the model footprint and datasets.

### Methods

#### `dashboard()`
Opens the interactive HTML dashboard in the default system web browser. 

#### `save_html(path: str)`
Renders and saves the self-contained HTML dashboard to the specified path.

#### `save_json(path: str)`
Serializes the entire report to a JSON file.

#### `save_markdown(path: str)`
Exports a human-readable markdown summary.

#### `save_csv(path: str)`
Exports a CSV file of the findings (dimension, title, severity).

#### `to_dataframe() -> pandas.DataFrame`
Returns the findings as a pandas DataFrame.
