$scripts = @(
    "01_basic_classification.py",
    "02_basic_regression.py",
    "05_json_export.py",
    "09_prescription_engine.py",
    "11_model_comparison.py",
    "13_data_leakage_detection.py",
    "14_overfitting_analysis.py",
    "08_custom_doctor.py",
    "12_probability_calibration.py",
    "15_production_readiness.py"
)

$passed = 0
$failed = 0
$errors = @()

foreach ($script in $scripts) {
    Write-Host "Running $script..." -ForegroundColor Cyan
    $output = python -W ignore $script 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  PASS" -ForegroundColor Green
        $passed++
    } else {
        Write-Host "  FAIL" -ForegroundColor Red
        $failed++
        $errors += "$script`: $($output | Select-Object -Last 5 | Out-String)"
    }
}

Write-Host "`n--- Results: $passed passed, $failed failed ---"
if ($errors.Count -gt 0) {
    Write-Host "Failures:"
    $errors | ForEach-Object { Write-Host $_ }
}
