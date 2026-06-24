# Test Report

## Summary

Added a pytest suite for the PySpark weather-prediction project focused on reproducible artifacts and cloud-output helper behavior. The tests avoid starting Spark or accessing Google Cloud by validating checked-in results and mocking GCS uploads.

## What Is Covered

- Model metric JSON files are valid and contain sane RMSE/R2 values.
- Reported model ranking is consistent:
  - GBT beats Random Forest.
  - Random Forest beats Linear Regression.
- Feature-importance CSV files are sorted descending.
- Feature importances are normalized to approximately 1.0.
- GCS helper functions upload text and plot payloads with the expected object names and content types.

## Why This Matters

The full training pipeline requires Spark, GCS, and a large weather dataset. These tests verify the project artifacts that communicate model quality, and they check the cloud-output boundary without needing real cloud credentials.

## Verification

Command:

```powershell
python -m pytest -q
```

Result:

```text
3 passed
```

## Files Added

- `tests/test_artifacts_and_helpers.py`
