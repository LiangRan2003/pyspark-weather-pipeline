
# Weather Temperature Prediction (Spark on Google Cloud Dataproc)

This repo contains the end‑to‑end pipeline for predicting **hourly temperature** from weather features using **Apache Spark (PySpark)** on **Google Cloud Dataproc**.  
**Single script only**: `code/weather_prediction_v6.py` — it performs **data loading → cleaning/feature engineering → model training (LR / RF / GBT) → evaluation → artifact export** in one run. *Intermediate cleaned data is **not** persisted to GCS; the script cleans in‑memory per job.*

---

## 0) Prepare the Dataset on GCS (untar first)

We **first upload** the raw archive `2024.tar.gz` to the Cloud Storage bucket via the **Console** (web UI).  
Because Spark/Dataproc can read individual `.gz` files but **not** a whole **`.tar.gz`** archive, we **extract** the tarball on the Dataproc master and then copy the extracted **CSV** files back to GCS.

**Exactly what we run (with our real bucket/folder names):**

```bash
# Our bucket name (actual)
BUCKET="mllib_for_tmp"

# Work dir on the master VM
WORK="/tmp/isd"
mkdir -p "$WORK"

# Stream‑extract the tar.gz from GCS into the VM, then upload the extracted files back to GCS
gsutil cat gs://$BUCKET/2024.tar.gz | tar -xz -C "$WORK"

# (Optional) Quick check of the extracted layout
find "$WORK" -maxdepth 2 -type f | head

# Copy all extracted files/subfolders (CSVs) back to GCS under data/ISD_2024/ (actual)
gsutil -m cp -r "$WORK"/* gs://$BUCKET/data/ISD_2024/
```

After this, you can verify under **Cloud Storage → `mllib_for_tmp` → `data/ISD_2024/`** that the CSVs are present.

---

## 1) Create a Dataproc Cluster (Console, point‑and‑click)

**Console path:** Dataproc → **Clusters** → **Create cluster**

Use the following settings (others default):

- **Region:** `asia-southeast1` (Singapore)  
- **Cluster type:** *Standard*  
- **Cluster mode:** *Single node*  
- **Image version:** *2.2 (Debian 12, Spark 3.x)*  
- **Primary node:** `n2-standard-16`, **300 GB** boot disk  
- *(Optional)* Components: **Jupyter**

Click **Create** and wait for **Running** status.
---

## 2) Submission Command (exact used for the assignment)

```bash
gcloud dataproc jobs submit pyspark gs://mllib_for_tmp/scripts/weather_prediction_v6.py \
  --cluster=cluster-f66f \
  --region=asia-southeast1 \
  --properties=spark.driver.memory=16g,spark.executor.memory=16g,spark.driver.maxResultSize=8g
```
If needed, upload the script first:
```bash
gsutil cp ./code/weather_prediction_v6.py gs://mllib_for_tmp/scripts/
```

---

## 3) Cluster & Environment

- **Project**: `project-2-476006`
- **Region**: `asia-southeast1`  (**Zone**: `asia-southeast1-b`)
- **Cluster**: `cluster-f66f` (single-node)
- **Image**: `2.2.69-debian12` (Spark 3.x)
- **Primary**: `n2-standard-16`, boot disk `pd-standard`, **300 GB**
- **Staging bucket**: `dataproc-staging-asia-southeast1-984949519657-2pv9urwq`
- **Memory flags used**: `spark.driver.memory=16g`, `spark.executor.memory=16g`, `spark.driver.maxResultSize=8g`

> A banner warning about service‑account permissions may appear; the cluster ran jobs successfully in our setup.

---

## 4) Data

- Source: 2024 hourly weather/ISD (CSV/Parquet on GCS).  
- After **Step 0**, the extracted CSVs live under: **`gs://mllib_for_tmp/data/ISD_2024/`** (actual)  
- Split: **70% train / 30% test** (fixed seed).  
- **Note**: Cleaned/processed data **is not written back** to GCS; cleaning happens in‑memory.

---

## 5) Pipeline (what the script does)

1. Load data from GCS.  
2. Clean & engineer features (null handling, casting, `VectorAssembler`, optional standardization).  
3. Split 70/30.  
4. Train **Linear Regression**, **Random Forest**, **GBT** with simple grids.  
5. Validate (2‑fold CV or TV split), select best by validation **RMSE**.  
6. Evaluate on test set and export artifacts.  

**Artifacts saved to GCS (from logs):**
- Figures & tables:  
  - `gs://mllib_for_tmp/results/lr_residual_plot.png`  
  - `gs://mllib_for_tmp/results/linear_regression_coefficients.csv`  
  - `gs://mllib_for_tmp/results/random_forest_importances.csv`  
  - `gs://mllib_for_tmp/results/gbt_importances.csv`
- **Best model** (GBT) directory:  
  - `gs://mllib_for_tmp/models/best_gbt_weather_model_final/`

---

## 6) Final Metrics (Test Set)

| Model | Selection | Best Params | RMSE | R² |
|---|---|---|---:|---:|
| **Linear Regression** | 2‑fold CV | `regParam=0.01`, `elasticNetParam=0.00` | **5.1924** | **0.8292** |
| **Random Forest** | 2‑fold CV | `numTrees=50`, `maxDepth=<bound method>`, `featureSubsetStrategy=auto` | **3.6129** | **0.9173** |
| **Gradient‑Boosted Trees** | 2‑fold CV | `maxDepth=<bound method>`, `numTrees=50`, `maxIter=<bound method>`, `numFeatures=12` | **3.4525** | **0.9245** |

> Some Spark UI logs display `<bound method ...>` for estimator attributes; effective values come from the grid. GBT is the best model and is exported.

---

## 7) How to Re‑Evaluate (load the trained model)

```python
from pyspark.ml import PipelineModel
model = PipelineModel.load("gs://mllib_for_tmp/models/best_gbt_weather_model_final")
# model.transform(test_df) → evaluate RMSE/R2
```

---

## 8) Local Copies of Artifacts (optional)

```bash
# GCS → local
gsutil -m cp -r gs://mllib_for_tmp/results ./results_local
gsutil -m cp -r gs://mllib_for_tmp/models/best_gbt_weather_model_final ./model_local
```

---

## 9) Hyperparameter Grids

- **LR**: `regParam ∈ {0.0, 0.01, 0.1}`, `elasticNetParam ∈ {0.0, 0.5, 1.0}`  
- **RF**: `numTrees ∈ {50, 100, 300}`, `maxDepth ∈ {8, 12, 16}`, `featureSubsetStrategy ∈ {"auto","sqrt"}`  
- **GBT**: `maxIter ∈ {100, 200}`, `maxDepth ∈ {4, 6, 8}`, `stepSize ∈ {0.05, 0.1}`

---

## 10) Troubleshooting

- Ensure region/cluster names match: `--region=asia-southeast1`, `--cluster=cluster-f66f`.  
- Enable Cloud Resource Manager API and confirm IAM for Dataproc service accounts if creation fails.  
- Avoid `collect()` on big DataFrames; keep memory flags above.  
- Use the Job link to inspect driver/executor logs.
