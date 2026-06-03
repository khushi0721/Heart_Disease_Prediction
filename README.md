# ❤️ Heart Disease Prediction using Machine Learning

### Predicting cardiovascular disease risk through multiple machine learning models

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-102230?style=for-the-badge)
![Open Source](https://img.shields.io/badge/Open%20Source-3DA639?style=for-the-badge&logo=opensourceinitiative&logoColor=white)

## Project Overview

This project uses machine learning to predict whether a patient is likely to have heart disease based on clinical attributes such as age, chest pain type, cholesterol level, resting blood pressure, maximum heart rate, exercise induced angina, and related medical measurements.

Heart disease prediction matters because early risk signals can help healthcare teams prioritize follow-up checks and make better use of clinical data. The objective of this notebook is to turn a structured heart disease dataset into a classification workflow, compare several supervised learning models, and identify the model with the best saved accuracy in the notebook.

The project is built as an end-to-end machine learning notebook. It covers data loading, missing value checks, exploratory analysis, categorical encoding, feature scaling, model training, model comparison, and a first attempt at model serialization with pickle.

## Key Features

- 🧹 Data cleaning checks with missing value analysis
- 📊 Exploratory data analysis with correlation and distribution plots
- 🧪 Class distribution review before model training
- 🔢 One-hot encoding for categorical medical features
- 📏 Feature scaling with `StandardScaler`
- 🤖 Model training with KNN, SVC, Decision Tree, and Random Forest
- 📈 Hyperparameter comparison for each model family
- 🏆 Best model selection based on notebook accuracy
- 💾 Pickle export attempt for a trained Support Vector Classifier

## Dataset Overview

The repository contains two CSV files:

| File | Rows | Columns | Notes |
| --- | ---: | ---: | --- |
| `train.csv` | 1,024 | 14 | Full training-style dataset included in the repository |
| `test.csv` | 303 | 14 | Dataset loaded and used by the notebook workflow |

The notebook reads `test.csv` into the `dataset` variable and then creates its own train/test split from those 303 rows.

### Columns From `train.csv`

| Feature | Description |
| --- | --- |
| `age` | Patient age |
| `sex` | Patient sex encoded as a numeric category |
| `cp` | Chest pain type encoded as a numeric category |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar encoded as a binary category |
| `restecg` | Resting electrocardiographic result encoded as a numeric category |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise induced angina encoded as a binary category |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels encoded as a numeric category |
| `thal` | Thalassemia result encoded as a numeric category |
| `target` | Prediction target. `0` means no heart disease and `1` means heart disease |

### `train.csv` Target Distribution

| Target | Count | Percentage |
| --- | ---: | ---: |
| `0` | 498 | 48.63% |
| `1` | 526 | 51.37% |

### `test.csv` Target Distribution Used In Notebook

| Target | Count | Percentage |
| --- | ---: | ---: |
| `0` | 138 | 45.54% |
| `1` | 165 | 54.46% |

Both CSV files have 13 input features and 1 target column. No missing values were found in either file.

## Machine Learning Pipeline

```text
CSV Dataset
     ↓
Data Inspection
     ↓
Missing Value Analysis
     ↓
Exploratory Data Analysis
     ↓
One-Hot Encoding
     ↓
Feature Scaling
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Serialization Attempt
```

### Notebook Workflow Details

| Step | Actual Implementation |
| --- | --- |
| Dataset loaded | `pd.read_csv('test.csv')` |
| Missing value check | `dataset.info()`, `missingno.bar()`, `missingno.matrix()` |
| EDA | Correlation matrix, histograms, target class bar chart |
| Categorical encoding | `pd.get_dummies()` on `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal` |
| Scaled columns | `age`, `trestbps`, `chol`, `thalach`, `oldpeak` |
| Scaler | `StandardScaler` |
| Split | `train_test_split(X, y, test_size=0.33, random_state=0)` |
| Feature count after encoding | 30 model input features |
| Serialized model | `svc_classifier.pkl` |

## Exploratory Data Analysis

The notebook includes a clear first pass through the dataset before training.

- The notebook dataset has 303 rows and 14 columns.
- There are no null values in the notebook dataset.
- The target classes are close enough in size to continue without resampling.
- The feature scales vary widely. For example, `age` ranges from 29 to 77 while `chol` ranges from 126 to 564 in `test.csv`.
- Histograms show that numeric and categorical columns have different distributions, which supports the decision to scale continuous features.
- The correlation matrix shows both positive and negative relationships with `target`.

### Strongest Correlations With `target` In `test.csv`

| Feature | Correlation |
| --- | ---: |
| `cp` | 0.434 |
| `thalach` | 0.422 |
| `slope` | 0.346 |
| `exang` | -0.437 |
| `oldpeak` | -0.431 |
| `ca` | -0.392 |
| `thal` | -0.344 |
| `sex` | -0.281 |

These values come from the dataset used in the notebook. They show that chest pain type, maximum heart rate, exercise induced angina, ST depression, major vessels, and thalassemia values are important signals for the classification task.

## Models Implemented

| Model | Description | Hyperparameters Tested |
| --- | --- | --- |
| K Neighbors Classifier | Distance-based classifier tested across different neighbor counts | `n_neighbors` from 1 to 20 |
| Support Vector Classifier | Margin-based classifier tested with multiple kernels | `linear`, `poly`, `rbf`, `sigmoid` |
| Decision Tree Classifier | Tree-based classifier tested with different feature limits | `max_features` from 1 to 30, `random_state=0` |
| Random Forest Classifier | Ensemble classifier tested with different forest sizes | `n_estimators` of 10, 100, 200, 500, 1000 with `random_state=0` |

## Model Performance

| Model | Best Notebook Accuracy | Best Setting |
| --- | ---: | --- |
| K Neighbors Classifier | 87.0% | `n_neighbors=8` |
| Random Forest Classifier | 84.0% | `n_estimators=100` or `500` |
| Support Vector Classifier | 83.0% | `kernel='linear'` |
| Decision Tree Classifier | 79.0% | `max_features=2`, `4`, or `18` |

🏆 **Best Performing Model: K Neighbors Classifier**

K Neighbors Classifier achieved the highest saved notebook score at 87.0% with 8 neighbors. The notebook compares K values from 1 to 20 and records the best score at `k=8`.

The Random Forest model came next with 84.0%, followed by the linear SVC at 83.0%. Decision Tree reached 79.0% in the saved output.

## Visualizations

The notebook renders these plots inline:

## Missing Value Bar Chart

![Missing Value Bar Chart](images/missing-value-bar-chart.png)

## Missing Value Matrix

![Missing Value Matrix](images/missing-value-matrix.png)

## Correlation Matrix

![Correlation Matrix](images/correlation-matrix.png)

## Feature Histograms

![Feature Histograms](images/feature-histograms.png)

## Target Class Distribution

![Target Class Distribution](images/target-class-distribution.png)

## KNN Score Comparison

![KNN Score Comparison](images/knn-score-comparison.png)

## SVC Kernel Comparison

![SVC Kernel Comparison](images/svc-kernel-comparison.png)

## Decision Tree Max Features Comparison

![Decision Tree Max Features Comparison](images/decision-tree-max-features-comparison.png)

## Random Forest Estimator Comparison

![Random Forest Estimator Comparison](images/random-forest-estimator-comparison.png)

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Missingno](https://img.shields.io/badge/Missingno-333333?style=flat-square)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter_Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)

## Repository Structure

```text
Heart_Disease_Prediction/
├── Heart_Disease_Prediction.ipynb
├── LICENSE
├── README.md
├── test.csv
└── train.csv
```

## Installation

Clone the repository:

```bash
git clone https://github.com/khushi0721/Heart_Disease_Prediction.git
cd Heart_Disease_Prediction
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn missingno scikit-learn jupyter
```

## Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Heart_Disease_Prediction.ipynb
```

Run the notebook cells from top to bottom to reproduce the analysis and model comparison saved in the notebook.

## Notes From The Current Notebook

- The notebook loads `test.csv`, even though the repository also includes `train.csv`.
- The final pickle step saves the last trained SVC model as `svc_classifier.pkl`.
- The final prediction cell passes a one-feature input array into a model that expects 30 encoded features, so that prediction cell raises a shape mismatch error.
- A production prediction path should apply the same dummy encoding and scaling steps used during training before calling `predict()`.

## Future Improvements

- Train and evaluate with a clearer train/validation/test structure
- Add a reusable preprocessing pipeline with `ColumnTransformer`
- Save the scaler, encoder, and model together with a single inference pipeline
- Add XGBoost and LightGBM for stronger tabular baselines
- Add SHAP explainability for model interpretation
- Add precision, recall, F1 score, ROC AUC, and confusion matrices
- Export notebook plots into an `images/` folder for README rendering
- Add FastAPI deployment for real-time predictions
- Add Docker support for repeatable local runs
- Add AWS deployment using EC2 as noted in the notebook
- Add CI/CD checks for notebook execution and linting

## What This Project Demonstrates

This project shows the core skills expected in an entry-level machine learning or data science portfolio project. It starts with a real tabular healthcare dataset, checks the data quality, studies feature distributions, transforms categorical columns, scales numeric columns, trains multiple classifiers, and compares their accuracy.

For software engineering roles, it shows comfort with Python, notebooks, structured data, and reproducible workflows. For machine learning roles, it shows the ability to move from raw CSV data to model evaluation while making practical preprocessing choices. For internships and startup roles, it shows initiative and the ability to turn a dataset into a working ML experiment.

The next step would be to convert the notebook workflow into a clean training script and inference pipeline. That would make the project easier to test, deploy, and maintain.

## Additional Deliverables

### Repository Description

Heart disease prediction with EDA, preprocessing, model comparison, and supervised machine learning in Python.

### GitHub Topics

```text
machine-learning
heart-disease-prediction
healthcare-ai
classification
data-science
python
pandas
numpy
scikit-learn
jupyter-notebook
exploratory-data-analysis
predictive-analytics
medical-data
supervised-learning
```

### Professional Banner Idea

Create a clean banner with a dark navy background, a simple ECG line, a small heart icon, and the title "Heart Disease Prediction using Machine Learning". Keep the layout minimal and use one accent color such as red or teal.

### Improvements To Stand Out For Recruiters

- Add a `requirements.txt` file
- Add an `images/` folder with exported notebook plots
- Refactor preprocessing and training into Python scripts
- Add a saved end-to-end sklearn pipeline for inference
- Add model evaluation beyond accuracy
- Add a FastAPI app with a `/predict` endpoint
- Add a small web form for entering patient features
- Add tests for preprocessing and prediction input shape
- Add Docker support
- Add a short project demo GIF

### README Score

| Version | Score | Reason |
| --- | ---: | --- |
| Before | 2/10 | The old README only had a title and one sentence. It did not explain the dataset, workflow, models, metrics, or how to run the project. |
| After | 8/10 | The new README documents the actual files, dataset shape, preprocessing, EDA, model comparison, notebook results, limitations, setup steps, and recruiter-facing project value. |

## Footer

⭐ If you found this project useful, consider giving it a star.

Built with Python, Machine Learning, and a passion for solving real-world healthcare challenges.
