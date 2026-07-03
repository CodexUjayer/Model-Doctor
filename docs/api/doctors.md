# Doctors API

## `BaseDoctor`

The abstract base class that all Doctors must implement.

```python
class BaseDoctor(ABC):
    name: str
    dimension: str
    priority: int

    @abstractmethod
    def examine(self, context: EvaluationContext) -> Diagnosis:
        pass
```

### Built-in Doctors

- `OverfittingDoctor`: Evaluates train/test memorization gap.
- `LeakageDoctor`: Evaluates proxy feature leakage.
- `PredictionDoctor`: Evaluates basic f1/accuracy/r2 sanity checks.
- `DataDoctor`: Evaluates row/column integrity, missing values, and class balance.
- `FeatureDoctor`: Evaluates dimensionality and variance.
- `CalibrationDoctor`: Evaluates probability reliability (Expected Calibration Error).
- `ProductionDoctor`: Evaluates serialized model size and inference latency.
- `GeneralizationDoctor`: Evaluates cross-validation fold stability.
