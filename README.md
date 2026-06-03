# Heart Disease Prediction

A small machine learning project for predicting heart disease from clinical tabular data.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

## Overview

This repository explores a heart disease dataset and compares several classification models. The target column is `target`, where `0` represents no heart disease and `1` represents heart disease.

The project includes both the original notebook workflow and a small Python package for repeatable training and prediction.

## Repository Contents

```text
Heart_Disease_Prediction/
|-- data/
|   |-- test.csv
|   `-- train.csv
|-- images/
|   |-- correlation-matrix.png
|   |-- decision-tree-max-features-comparison.png
|   |-- feature-histograms.png
|   |-- knn-score-comparison.png
|   |-- missing-value-bar-chart.png
|   |-- missing-value-matrix.png
|   |-- random-forest-estimator-comparison.png
|   |-- svc-kernel-comparison.png
|   `-- target-class-distribution.png
|-- notebooks/
|   `-- heart_disease_prediction.ipynb
|-- src/
|   |-- __init__.py
|   |-- predict.py
|   |-- preprocess.py
|   `-- train.py
|-- tests/
|   `-- test_preprocess.py
|-- .gitignore
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## Dataset

The repository includes two CSV files with the same 14 columns.

| File | Rows | Columns |
| --- | ---: | ---: |
| `data/train.csv` | 1,024 | 14 |
| `data/test.csv` | 303 | 14 |

Both files contain 13 input features and 1 target column. No missing values were found in either file.

The notebook source mentions Kaggle dataset `ronitf/heart-disease-uci`.

## Columns

| Column | Description |
| --- | --- |
| `age` | Patient age |
| `sex` | Sex encoded as a numeric category |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar |
| `restecg` | Resting electrocardiographic result |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise induced angina |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels |
| `thal` | Thalassemia result |
| `target` | Classification target. `0` means no heart disease and `1` means heart disease |

## Target Distribution

### `data/train.csv`

| Target | Count | Percentage |
| --- | ---: | ---: |
| `0` | 498 | 48.63% |
| `1` | 526 | 51.37% |

### `data/test.csv`

| Target | Count | Percentage |
| --- | ---: | ---: |
| `0` | 138 | 45.54% |
| `1` | 165 | 54.46% |

## Workflow

```text
Load CSV data
  |
Validate feature columns
  |
Encode categorical features
  |
Scale numeric features
  |
Train classification models
  |
Compare evaluation metrics
  |
Save the best sklearn pipeline
  |
Run predictions with the saved pipeline
```

## Preprocessing

The training script uses a `ColumnTransformer`.

Numeric features are scaled with `StandardScaler`:

```text
age, trestbps, chol, thalach, oldpeak
```

Categorical features are one-hot encoded with `OneHotEncoder(handle_unknown="ignore")`:

```text
sex, cp, fbs, restecg, exang, slope, ca, thal
```

## Exploratory Analysis

The notebook includes missing value plots, a correlation matrix, histograms for all columns, and a target class distribution chart.

In `data/test.csv`, the strongest positive correlations with `target` are:

| Feature | Correlation |
| --- | ---: |
| `cp` | 0.434 |
| `thalach` | 0.422 |
| `slope` | 0.346 |

The strongest negative correlations with `target` are:

| Feature | Correlation |
| --- | ---: |
| `exang` | -0.437 |
| `oldpeak` | -0.431 |
| `ca` | -0.392 |
| `thal` | -0.344 |
| `sex` | -0.281 |

## Models

The notebook compares:

| Model | Parameters tested in notebook |
| --- | --- |
| K Neighbors Classifier | `n_neighbors` from 1 to 20 |
| Support Vector Classifier | `linear`, `poly`, `rbf`, and `sigmoid` kernels |
| Decision Tree Classifier | `max_features` from 1 to 30 with `random_state=0` |
| Random Forest Classifier | `n_estimators` of 10, 100, 200, 500, and 1000 with `random_state=0` |

The Python training script compares the same model families with fixed settings selected from the notebook:

| Model key | Implementation |
| --- | --- |
| `knn` | `KNeighborsClassifier(n_neighbors=8)` |
| `svc_linear` | `SVC(kernel="linear", probability=True, random_state=0)` |
| `decision_tree` | `DecisionTreeClassifier(max_features=18, random_state=0)` |
| `random_forest` | `RandomForestClassifier(n_estimators=100, random_state=0)` |

## Notebook Results

The scores below are the accuracy values saved in the notebook output.

| Model | Best accuracy | Setting |
| --- | ---: | --- |
| K Neighbors Classifier | 87.0% | `n_neighbors=8` |
| Random Forest Classifier | 84.0% | `n_estimators=100` or `500` |
| Support Vector Classifier | 83.0% | `kernel='linear'` |
| Decision Tree Classifier | 79.0% | `max_features=2`, `4`, or `18` |

K Neighbors Classifier has the highest saved score in the notebook at 87.0%.

## Plots

### Missing Value Bar Chart

![Missing Value Bar Chart](images/missing-value-bar-chart.png)

### Missing Value Matrix

![Missing Value Matrix](images/missing-value-matrix.png)

### Correlation Matrix

![Correlation Matrix](images/correlation-matrix.png)

### Feature Histograms

![Feature Histograms](images/feature-histograms.png)

### Target Class Distribution

![Target Class Distribution](images/target-class-distribution.png)

### KNN Score Comparison

![KNN Score Comparison](images/knn-score-comparison.png)

### SVC Kernel Comparison

![SVC Kernel Comparison](images/svc-kernel-comparison.png)

### Decision Tree Max Features Comparison

![Decision Tree Max Features Comparison](images/decision-tree-max-features-comparison.png)

### Random Forest Estimator Comparison

![Random Forest Estimator Comparison](images/random-forest-estimator-comparison.png)

## Setup

Clone the repository:

```bash
git clone https://github.com/khushi0721/Heart_Disease_Prediction.git
cd Heart_Disease_Prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Notebook

```bash
jupyter notebook notebooks/heart_disease_prediction.ipynb
```

## Training From Python

```bash
python src/train.py --data data/train.csv
```

This writes:

```text
models/heart_disease_pipeline.joblib
reports/model_metrics.csv
```

Those generated files are ignored by git.

## Running Predictions

Train the pipeline first:

```bash
python src/train.py --data data/train.csv
```

Then run predictions on a CSV file with the same feature columns:

```bash
python src/predict.py --input data/test.csv
```

To save predictions:

```bash
python src/predict.py --input data/test.csv --output reports/predictions.csv
```

## Running Tests

```bash
pytest
```

## Current Limitations

- The notebook still reports accuracy only. The Python training script adds more metrics.
- The saved notebook outputs come from the original notebook run.
- The project does not include a web app or API.
- Model artifacts are generated locally and are not committed.

## Roadmap

- Add confusion matrix and ROC curve plots from the Python training script
- Add cross-validation for model comparison
- Add a small API for prediction
- Add Docker support for reproducible local runs
- Add a short example input file for inference

## License

This project is licensed under the terms in the [LICENSE](LICENSE) file.
