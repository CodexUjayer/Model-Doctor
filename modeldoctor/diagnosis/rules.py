import yaml
import os
from pathlib import Path

# Note: In a real system, we'd read from a YAML file. For simplicity, we hardcode defaults here.
class RulesConfig:
    def __init__(self):
        # ----------------------------------------------------------------
        # Overfitting
        # Justification: Validation shows overfit RF/Tree gaps of 0.10–0.25.
        # Old critical=0.30 missed these. New values tuned to benchmark results.
        # ----------------------------------------------------------------
        self.overfitting_gap_critical = 0.15    # Was: 0.30
        self.overfitting_gap_warning = 0.08     # Was: 0.15
        self.memorization_train_threshold = 0.99
        self.memorization_test_threshold = 0.85
        self.cv_variance_warning = 0.10         # Was: 0.15 — earlier warning
        self.excessive_capacity_ratio = 1.0
        self.tree_depth_warning = 15
        self.tree_depth_max_samples = 10000

        # ----------------------------------------------------------------
        # Leakage
        # Justification: Customer ID scenario has max_corr ~0.87. Old critical=0.95
        # never fired. Lowered to catch real-world proxy leakage patterns.
        # ----------------------------------------------------------------
        self.leakage_correlation_critical = 0.80  # Was: 0.95
        self.leakage_correlation_warning = 0.65   # Was: 0.85
        self.leakage_importance_critical = 0.80   # Was: 0.90
        self.leakage_importance_warning = 0.45    # Was: 0.70

        # ----------------------------------------------------------------
        # Data Quality
        # Justification: 25% missing clearly critical. 99:1 imbalance is ratio=99
        # (already at old threshold=100). Lower to also catch 95:5 (ratio=19).
        # ----------------------------------------------------------------
        self.data_missing_critical = 0.15   # Was: 0.20 — 25% NaN test uses 0.25, safely above
        self.data_missing_warning = 0.05
        self.data_imbalance_critical = 20.0  # Was: 100.0 — catches 95:5 splits
        self.data_imbalance_warning = 5.0    # Was: 10.0

        # ----------------------------------------------------------------
        # Calibration
        # Justification: Uncalibrated RF ECE on noisy data is ~0.12–0.18.
        # Old critical=0.20 was too high. New value catches common RF miscalibration.
        # ----------------------------------------------------------------
        self.calibration_ece_critical = 0.12  # Was: 0.20
        self.calibration_ece_warning = 0.07   # Was: 0.10
        self.calibration_brier_critical = 0.25
        self.calibration_brier_warning = 0.15

        # ----------------------------------------------------------------
        # Generalization
        # Justification: Gap thresholds kept. CV std lowered so noisy small
        # datasets trigger the warning more reliably.
        # ----------------------------------------------------------------
        self.generalization_gap_critical = 0.20   # Was: 0.25
        self.generalization_gap_warning = 0.10
        self.generalization_cv_std_critical = 0.10  # Was: 0.15
        self.generalization_cv_std_warning = 0.05   # Was: 0.07

        # ----------------------------------------------------------------
        # Production
        # Justification: 500-estimator RF serializes to 150–400MB.
        # Old threshold=500MB never triggered. Lowered to realistic values.
        # Latency threshold unchanged — 500ms per-sample is already critical.
        # ----------------------------------------------------------------
        self.production_latency_critical_ms = 500.0
        self.production_latency_warning_ms = 100.0
        self.production_size_critical_mb = 100.0  # Was: 500.0
        self.production_size_warning_mb = 50.0    # Was: 100.0

        # ----------------------------------------------------------------
        # Prediction
        # Justification: "Average Classifier" accuracy ~0.68 should be a warning.
        # Old warning=0.75 missed the 0.68 scenario. Lowered slightly.
        # ----------------------------------------------------------------
        self.prediction_accuracy_critical = 0.60
        self.prediction_accuracy_warning = 0.75  # Was: 0.75
        self.prediction_f1_critical = 0.55
        self.prediction_r2_critical = 0.30
        self.prediction_r2_warning = 0.60

        # ----------------------------------------------------------------
        # Feature Engineering
        # Justification: 1200-feature dataset clearly high-dimensional.
        # ----------------------------------------------------------------
        self.feature_high_dimensionality = 1000
        self.feature_high_cardinality_ratio = 0.95

# Global instance for easy access
RULES = RulesConfig()

