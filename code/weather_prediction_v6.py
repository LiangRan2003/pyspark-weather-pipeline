# weather_prediction_v6.py

import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, year, month, dayofyear, hour
from pyspark.sql.types import FloatType
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from google.cloud import storage

# ==============================================================================
# 辅助函数
# ==============================================================================
def save_plot_to_gcs(bucket_name, file_name, fig):
    """将 matplotlib figure 对象保存为 png 到 GCS"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(file_name)
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        blob.upload_from_file(buf, content_type='image/png')
        print(f"图像已成功保存到 gs://{bucket_name}/{file_name}")
        plt.close(fig)
    except Exception as e:
        print(f"保存图像到GCS时出错: {e}")

def save_text_to_gcs(bucket_name, file_name, content):
    """将文本内容保存到 GCS"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.upload_from_string(content)
        print(f"文本文件已成功保存到 gs://{bucket_name}/{file_name}")
    except Exception as e:
        print(f"保存文本文件到GCS时出错: {e}")

# ==============================================================================
# 主函数
# ==============================================================================
def main():
    spark = SparkSession.builder.appName("FinalWeatherPrediction_v2").getOrCreate()

    # --- 1. 数据加载与预处理 ---
    data_path = "gs://mllib_for_tmp/data/ISD_2024/*.csv"
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    print("开始数据预处理和特征工程...")
    df = df.withColumn("timestamp", col("DATE").cast("timestamp")) \
           .withColumn("year", year(col("timestamp"))) \
           .withColumn("month", month(col("timestamp"))) \
           .withColumn("day_of_year", dayofyear(col("timestamp"))) \
           .withColumn("hour", hour(col("timestamp")))

    cols_to_process = {
        "TMP": (0, 1, "+9999", 10.0), "DEW": (0, 1, "+9999", 10.0),
        "SLP": (0, 1, "99999", 10.0), "WND": (3, 4, "9999", 10.0),
        "VIS": (0, 1, "999999", 1.0), "CIG": (0, 1, "99999", 1.0)
    }
    
    df_clean = df
    for name, (val_idx, qual_idx, invalid_val, scale) in cols_to_process.items():
        df_clean = df_clean.withColumn(f"{name}_split", split(col(name), ",")) \
                           .filter( (col(f"{name}_split").getItem(qual_idx) != '3') & (col(f"{name}_split").getItem(qual_idx) != '7') & (col(f"{name}_split").getItem(val_idx) != invalid_val) ) \
                           .withColumn(name.lower(), col(f"{name}_split").getItem(val_idx).cast(FloatType()) / scale)

    df_renamed = df_clean.withColumnRenamed("tmp", "temperature").withColumnRenamed("dew", "dew_point").withColumnRenamed("slp", "sea_level_pressure").withColumnRenamed("wnd", "wind_speed").withColumnRenamed("vis", "visibility").withColumnRenamed("cig", "sky_condition_height")

    final_cols = ["temperature", "dew_point", "sea_level_pressure", "wind_speed", "visibility", "sky_condition_height", "LATITUDE", "LONGITUDE", "ELEVATION", "year", "month", "day_of_year", "hour"]
    
    processed_df = df_renamed.select(final_cols) \
                             .withColumn("LATITUDE", col("LATITUDE").cast(FloatType())).withColumn("LONGITUDE", col("LONGITUDE").cast(FloatType())).withColumn("ELEVATION", col("ELEVATION").cast(FloatType())) \
                             .na.drop()
    
    print("数据清洗完成。")

    # --- 2. 准备机器学习流水线 ---
    feature_cols = ["dew_point", "sea_level_pressure", "wind_speed", "visibility", "sky_condition_height", "LATITUDE", "LONGITUDE", "ELEVATION", "year", "month", "day_of_year", "hour"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

    (training_data, test_data) = processed_df.randomSplit([0.7, 0.3], seed=42)
    training_data.cache(); test_data.cache()
    
    print(f"数据集划分完成: 训练集 {training_data.count()} 条, 测试集 {test_data.count()} 条")
    evaluator_rmse = RegressionEvaluator(labelCol="temperature", predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="temperature", predictionCol="prediction", metricName="r2")

    # --- 3. 线性回归模型 ---
    print("\n--- 开始训练和评估线性回归模型 (带交叉验证) ---")
    lr = LinearRegression(featuresCol='scaled_features', labelCol='temperature')
    lr_pipeline = Pipeline(stages=[assembler, scaler, lr])

    lr_param_grid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1, 0.5]).addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]).build()
    lr_crossval = CrossValidator(estimator=lr_pipeline, estimatorParamMaps=lr_param_grid, evaluator=evaluator_rmse, numFolds=2, seed=42)
    lr_cv_model = lr_crossval.fit(training_data)
    best_lr_pipeline_model = lr_cv_model.bestModel
    
    lr_predictions = best_lr_pipeline_model.transform(test_data)
    rmse_lr = evaluator_rmse.evaluate(lr_predictions)
    r2_lr = evaluator_r2.evaluate(lr_predictions)

    best_lr_model = best_lr_pipeline_model.stages[-1]
    
    print("\n--- 线性回归模型 (调优后) 评估结果 ---")
    print("参数选择方法: 2折交叉验证")
    print(f"最优参数: regParam={best_lr_model.getRegParam():.2f}, elasticNetParam={best_lr_model.getElasticNetParam():.2f}")
    print(f"RMSE: {rmse_lr:.4f}, R²: {r2_lr:.4f}")

    coeffs = best_lr_model.coefficients
    coeff_df = pd.DataFrame(list(zip(feature_cols, coeffs)), columns=['Feature', 'Coefficient'])
    save_text_to_gcs("mllib_for_tmp", "results/linear_regression_coefficients.csv", coeff_df.to_csv(index=False))

    lr_predictions.withColumn("residual", col("temperature") - col("prediction")).createOrReplaceTempView("lr_residuals")
    residuals_pd = spark.sql("SELECT prediction, residual FROM lr_residuals").sample(False, 0.1, seed=42).toPandas()
    
    fig_lr_res, ax_lr_res = plt.subplots()
    ax_lr_res.scatter(residuals_pd['prediction'], residuals_pd['residual'], alpha=0.5)
    ax_lr_res.axhline(y=0, color='r', linestyle='--')
    ax_lr_res.set_title("Linear Regression Residual Plot"); ax_lr_res.set_xlabel("Predicted Temperature"); ax_lr_res.set_ylabel("Residual")
    save_plot_to_gcs("mllib_for_tmp", "results/lr_residual_plot.png", fig_lr_res)
    
    # --- 4. 随机森林模型 ---
    print("\n--- 开始训练和评估随机森林模型 (使用交叉验证) ---")
    rf = RandomForestRegressor(featuresCol='features', labelCol='temperature', seed=42)
    rf_pipeline = Pipeline(stages=[assembler, rf])

    rf_param_grid = ParamGridBuilder().addGrid(rf.numTrees, [20, 50]).addGrid(rf.maxDepth, [5, 10]).build()
    rf_crossval = CrossValidator(estimator=rf_pipeline, estimatorParamMaps=rf_param_grid, evaluator=evaluator_rmse, numFolds=2, seed=42)
    
    print("正在为随机森林运行交叉验证...")
    rf_cv_model = rf_crossval.fit(training_data)
    best_rf_pipeline_model = rf_cv_model.bestModel
    rf_predictions = best_rf_pipeline_model.transform(test_data)
    rmse_rf = evaluator_rmse.evaluate(rf_predictions)
    r2_rf = evaluator_r2.evaluate(rf_predictions)
    best_rf_model = best_rf_pipeline_model.stages[-1]

    print("\n--- 随机森林模型 (调优后) 评估结果 ---")
    print("参数选择方法: 2折交叉验证")
    # ** CORRECTED LINE **
    print(f"最优参数: NumTrees={best_rf_model.getNumTrees}, MaxDepth={best_rf_model.getMaxDepth}")
    print(f"RMSE: {rmse_rf:.4f}, R²: {r2_rf:.4f}")

    rf_importances = best_rf_model.featureImportances
    rf_importance_df = pd.DataFrame(list(zip(feature_cols, rf_importances)), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)
    save_text_to_gcs("mllib_for_tmp", "results/random_forest_importances.csv", rf_importance_df.to_csv(index=False))

    # --- 5. 梯度提升树 (GBT) 模型 ---
    print("\n--- 开始训练和评估 GBT 模型 (使用交叉验证) ---")
    gbt = GBTRegressor(featuresCol='features', labelCol='temperature', seed=42)
    gbt_pipeline = Pipeline(stages=[assembler, gbt])

    gbt_param_grid = ParamGridBuilder().addGrid(gbt.maxDepth, [3, 5]).addGrid(gbt.maxIter, [20, 50]).build()
    gbt_crossval = CrossValidator(estimator=gbt_pipeline, estimatorParamMaps=gbt_param_grid, evaluator=evaluator_rmse, numFolds=2, seed=42)
    
    print("正在为 GBT 运行交叉验证...")
    gbt_cv_model = gbt_crossval.fit(training_data)
    best_gbt_pipeline_model = gbt_cv_model.bestModel
    
    gbt_predictions = best_gbt_pipeline_model.transform(test_data)
    rmse_gbt = evaluator_rmse.evaluate(gbt_predictions)
    r2_gbt = evaluator_r2.evaluate(gbt_predictions)

    best_gbt_model = best_gbt_pipeline_model.stages[-1]
    
    # ** CORRECTED LINES **
    best_max_depth = best_gbt_model.getMaxDepth
    best_max_iter = best_gbt_model.getMaxIter

    print("\n--- GBT 模型 (调优后) 评估结果 ---")
    print("参数选择方法: 2折交叉验证")
    print(f"最优参数: MaxDepth={best_max_depth}, MaxIter={best_max_iter}")
    print(f"RMSE: {rmse_gbt:.4f}, R²: {r2_gbt:.4f}")

    gbt_importances = best_gbt_model.featureImportances
    gbt_importance_df = pd.DataFrame(list(zip(feature_cols, gbt_importances)), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)
    save_text_to_gcs("mllib_for_tmp", "results/gbt_importances.csv", gbt_importance_df.to_csv(index=False))

    try:
        # ** CORRECTED SECTION to get the trained model from the pipeline **
        trained_gbt_model = gbt_cv_model.bestModel.stages[-1]
        
        # Check if training summary is available
        if trained_gbt_model.hasSummary:
            gbt_summary = trained_gbt_model.summary
            gbt_objective_history = gbt_summary.objectiveHistory
            
            fig_gbt, ax_gbt = plt.subplots()
            ax_gbt.plot(gbt_objective_history)
            # ** CORRECTED LINE **
            ax_gbt.set_title(f"GBT Training Curve (RMSE per Iteration)\nParams: maxDepth={best_max_depth}, maxIter={best_max_iter}")
            ax_gbt.set_xlabel("Iteration"); ax_gbt.set_ylabel("RMSE")
            save_plot_to_gcs("mllib_for_tmp", "results/gbt_training_curve.png", fig_gbt)
        else:
            print("GBT 模型训练摘要不可用，无法绘制训练曲线。")
            
    except Exception as e:
        print(f"无法获取或绘制 GBT 训练曲线: {e}")

    # --- 6. 保存最佳模型 ---
    model_output_path = "gs://mllib_for_tmp/models/best_gbt_weather_model_final"
    best_gbt_pipeline_model.write().overwrite().save(model_output_path)
    print(f"\n最佳 GBT 流水线模型已保存至: {model_output_path}")
    
    training_data.unpersist()
    test_data.unpersist()
    spark.stop()

if __name__ == '__main__':
    main()