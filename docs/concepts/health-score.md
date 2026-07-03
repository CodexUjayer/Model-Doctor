# Health Score

The **health score** is ModelDoctor's single, unified measure of a model's overall quality. It is a float between 0 and 100, where 100 represents a model with no detected issues across any diagnostic dimension.

## How It Is Calculated

After all Doctors have run, the `HealthScorer` aggregates the individual `dimension_score` values from each Doctor's `Diagnosis`. These scores are weighted by the Doctor's assigned `priority` and by the severity of its findings.

The final health score accounts for:

- **Dimension scores**: Each Doctor assigns its examined dimension a score between 0.0 and 100.0.
- **Finding severity**: `CRITICAL` findings penalize the score more heavily than `WARNING` or `INFO` findings.
- **Doctor priority**: Higher-priority dimensions (e.g., leakage, overfitting) carry more weight.

## Score Grades

| Score | Grade | Meaning |
|---|---|---|
| 90–100 | A | Model is healthy and production-ready. |
| 75–89 | B | Minor issues detected. Review recommendations. |
| 60–74 | C | Moderate issues present. Address before deploying. |
| 40–59 | D | Significant problems detected. Not recommended for production. |
| 0–39 | F | Severe failures. Likely data leakage or total overfitting. |

## Reading the Score

```python
import modeldoctor as md

report = md.diagnose(model, X_train, y_train, X_test, y_test)

print(f"Health Score: {report.health_score.overall:.1f} / 100")
print(f"Grade:        {report.health_score.grade}")
```

## Confidence and Risk

The health score is derived from two intermediate scores calculated for each finding:

- **Confidence**: How statistically certain is ModelDoctor that this issue is real? (Low / Medium / High)
- **Risk**: How dangerous is this issue for the application? (Info / Low / Medium / High / Critical)

The [Confidence Engine](../concepts/evaluation-context.md) and the risk pipeline ensure that a mathematically certain but practically harmless phenomenon (e.g., a 0.01% generalization gap) does not generate a false `CRITICAL` alert.

## Interpreting Zero Findings

A health score of 100 does not guarantee model correctness — it means ModelDoctor found no evidence of the specific failure modes it checks. Always combine ModelDoctor results with domain-specific validation and business logic testing.
