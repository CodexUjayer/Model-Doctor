# Validation API

The Validation Laboratory is an independent module intended for developers contributing to the ModelDoctor core library. It is executed via the `benchmark_runner.py` script.

## `ValidationScenario`

The base class for all deterministic edge-case tests.

```python
class ValidationScenario(ABC):
    name: str
    category: str
    description: str
    random_seed: int

    @abstractmethod
    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        """Must return X_train, X_test, y_train, y_test"""
        pass

    @abstractmethod
    def build_model(self, X_train: Any, y_train: Any) -> Any:
        """Must return a fitted model"""
        pass

    @abstractmethod
    def expected(self) -> ExpectedResult:
        """Define the strict passing conditions"""
        pass
```

## `ExpectedResult`

Defines the exact assertions the runner will check against the generated `Report`.

```python
class ExpectedResult(BaseModel):
    passed: bool
    findings: Optional[List[str]] = None
    severity: Optional[List[str]] = None
```
