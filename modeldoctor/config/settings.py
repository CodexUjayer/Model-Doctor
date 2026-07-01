"""Configuration system for ModelDoctor.

All tuneable knobs live here. Consumers can override defaults by:

1. Passing a ``ModelDoctorConfig`` instance to ``diagnose()``.
2. Setting environment variables prefixed with ``MODELDOCTOR_``.
3. Loading from a YAML / JSON config file via ``ModelDoctorConfig.from_file()``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from modeldoctor.models.enums import ExplainabilityMode, ReportFormat


class ModelDoctorConfig(BaseSettings):
    """Runtime configuration for a ModelDoctor diagnostic run.

    All fields can be overridden via environment variables using the prefix
    ``MODELDOCTOR_``.  For example, set ``MODELDOCTOR_CV_FOLDS=10`` to change
    the default cross-validation fold count.

    Attributes:
        explainability: SHAP/LIME computation mode.
        cv_folds: Number of folds for cross-validation.
        cv_scoring: Scoring metric string for cross-validation.
        max_shap_samples: Maximum number of samples to use for SHAP computation.
        enable_parallel: Run independent doctors concurrently.
        max_workers: Maximum thread-pool workers when parallel is enabled.
        enabled_doctors: Whitelist of doctor names to run (empty = all).
        disabled_doctors: Doctor names to skip.
        output_formats: Report formats to generate automatically.
        output_dir: Directory to write reports to (``None`` = do not auto-save).
        report_title: Custom title for the generated report.
        importance_top_n: Number of top features to include in feature reviews.
        leakage_correlation_threshold: Pearson |r| above which a feature is
            flagged as potentially leaking the target.
        overfitting_gap_threshold: Train–test metric gap (0–1) above which
            overfitting is flagged as an error.
        overfitting_warning_threshold: Gap above which a warning is issued.
        production_latency_budget_ms: Maximum acceptable prediction latency in ms.
        custom_rules_dir: Path to a directory containing extra YAML rule files.
        log_level: Logging verbosity (``"DEBUG"`` | ``"INFO"`` | ``"WARNING"``).
        ai_review_provider: LLM provider for AI-enhanced reviews (``None`` = skip).
        ai_review_model: Model name to use with the LLM provider.
    """

    model_config = SettingsConfigDict(
        env_prefix="MODELDOCTOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Explainability ---
    explainability: ExplainabilityMode = ExplainabilityMode.AUTO

    # --- Cross-Validation ---
    cv_folds: int = Field(default=5, ge=2, le=20)
    cv_scoring: Optional[str] = None  # auto-selected based on task type

    # --- SHAP ---
    max_shap_samples: int = Field(default=500, ge=10)

    # --- Benchmarking ---
    compare_models: bool = False

    # --- Parallelism ---
    enable_parallel: bool = True
    max_workers: int = Field(default=4, ge=1)

    # --- Doctor selection ---
    enabled_doctors: List[str] = Field(default_factory=list)   # empty = all
    disabled_doctors: List[str] = Field(default_factory=list)

    # --- Reporting ---
    output_formats: List[ReportFormat] = Field(
        default_factory=lambda: [ReportFormat.MARKDOWN, ReportFormat.HTML]
    )
    output_dir: Optional[Path] = None
    report_title: str = "ModelDoctor Diagnostic Report"

    # --- Feature review ---
    importance_top_n: int = Field(default=20, ge=1)

    # --- Thresholds ---
    leakage_correlation_threshold: float = Field(default=0.95, ge=0.5, le=1.0)
    overfitting_gap_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    overfitting_warning_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    production_latency_budget_ms: float = Field(default=100.0, ge=1.0)

    # --- Custom rules ---
    custom_rules_dir: Optional[Path] = None

    # --- Logging ---
    log_level: str = "INFO"

    # --- AI review (optional Layer 2) ---
    ai_review_provider: Optional[str] = None   # "openai" | "anthropic" | None
    ai_review_model: Optional[str] = None

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got '{v}'.")
        return upper

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ModelDoctorConfig":
        """Load configuration from a YAML or JSON file.

        Args:
            path: Path to the config file (*.yaml, *.yml, or *.json).

        Returns:
            A new ``ModelDoctorConfig`` instance populated from the file.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the file extension is not recognised.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")

        if p.suffix in {".yaml", ".yml"}:
            data: Dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        elif p.suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
        else:
            raise ValueError(f"Unsupported config file format: '{p.suffix}'.")

        return cls(**data)
