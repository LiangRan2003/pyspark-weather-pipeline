import csv
import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_model_metric_artifacts_are_valid_and_ranked():
    metrics = {}
    for name, path in {
        "linear_regression": ROOT / "models" / "linear_regression" / "lr_metrics.json",
        "random_forest": ROOT / "models" / "random_forest" / "rf_metrics.json",
        "gbt": ROOT / "models" / "gbt" / "gbt_metrics.json",
    }.items():
        metrics[name] = json.loads(path.read_text())
        assert metrics[name]["rmse_test"] > 0
        assert 0 <= metrics[name]["r2_test"] <= 1
        assert metrics[name]["data_split"] == "70/30"

    assert metrics["gbt"]["rmse_test"] < metrics["random_forest"]["rmse_test"]
    assert metrics["random_forest"]["rmse_test"] < metrics["linear_regression"]["rmse_test"]


def test_feature_importance_artifacts_are_sorted_and_normalized():
    for path in [
        ROOT / "models" / "gbt" / "gbt_importances.csv",
        ROOT / "models" / "random_forest" / "random_forest_importances.csv",
    ]:
        rows = list(csv.DictReader(path.open(newline="")))
        values = [float(row["Importance"]) for row in rows]

        assert len(rows) >= 10
        assert values == sorted(values, reverse=True)
        assert abs(sum(values) - 1.0) < 1e-6


def test_gcs_helpers_upload_expected_payloads(monkeypatch):
    uploads = []

    class FakeBlob:
        def __init__(self, name):
            self.name = name

        def upload_from_string(self, content):
            uploads.append(("string", self.name, content))

        def upload_from_file(self, buf, content_type=None):
            uploads.append(("file", self.name, content_type, len(buf.read())))

    class FakeBucket:
        def blob(self, name):
            return FakeBlob(name)

    class FakeClient:
        def get_bucket(self, name):
            uploads.append(("bucket", name))
            return FakeBucket()

    storage_module = types.ModuleType("google.cloud.storage")
    storage_module.Client = FakeClient
    google_module = types.ModuleType("google")
    cloud_module = types.ModuleType("google.cloud")
    cloud_module.storage = storage_module
    google_module.cloud = cloud_module

    sys.modules["google"] = google_module
    sys.modules["google.cloud"] = cloud_module
    sys.modules["google.cloud.storage"] = storage_module

    pyspark_module = types.ModuleType("pyspark")
    pyspark_sql_module = types.ModuleType("pyspark.sql")
    pyspark_sql_module.SparkSession = object
    pyspark_sql_functions = types.ModuleType("pyspark.sql.functions")
    for name in ["split", "col", "year", "month", "dayofyear", "hour"]:
        setattr(pyspark_sql_functions, name, lambda *args, **kwargs: None)
    pyspark_sql_types = types.ModuleType("pyspark.sql.types")
    pyspark_sql_types.FloatType = object
    pyspark_ml = types.ModuleType("pyspark.ml")
    pyspark_ml.Pipeline = object
    pyspark_ml.PipelineModel = object
    pyspark_ml_feature = types.ModuleType("pyspark.ml.feature")
    pyspark_ml_feature.VectorAssembler = object
    pyspark_ml_feature.StandardScaler = object
    pyspark_ml_regression = types.ModuleType("pyspark.ml.regression")
    pyspark_ml_regression.LinearRegression = object
    pyspark_ml_regression.GBTRegressor = object
    pyspark_ml_regression.RandomForestRegressor = object
    pyspark_ml_evaluation = types.ModuleType("pyspark.ml.evaluation")
    pyspark_ml_evaluation.RegressionEvaluator = object
    pyspark_ml_tuning = types.ModuleType("pyspark.ml.tuning")
    pyspark_ml_tuning.ParamGridBuilder = object
    pyspark_ml_tuning.CrossValidator = object
    for module in [
        pyspark_module,
        pyspark_sql_module,
        pyspark_sql_functions,
        pyspark_sql_types,
        pyspark_ml,
        pyspark_ml_feature,
        pyspark_ml_regression,
        pyspark_ml_evaluation,
        pyspark_ml_tuning,
    ]:
        sys.modules[module.__name__] = module

    spec = importlib.util.spec_from_file_location(
        "weather_prediction_v6", ROOT / "code" / "weather_prediction_v6.py"
    )
    weather = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(weather)

    weather.save_text_to_gcs("bucket", "metrics.txt", "hello")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    weather.save_plot_to_gcs("bucket", "plot.png", fig)

    assert ("string", "metrics.txt", "hello") in uploads
    assert any(item[0:3] == ("file", "plot.png", "image/png") and item[3] > 0 for item in uploads)
