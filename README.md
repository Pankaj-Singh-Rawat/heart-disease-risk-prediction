# Heart Disease Risk Prediction

A machine learning classifier that predicts heart disease risk from patient demographic and clinical attributes, using the UCI Heart Disease (Cleveland) dataset.

## Problem

Early identification of heart disease risk from routine clinical measurements can support timely intervention. This project builds and compares multiple classification models to predict the presence of heart disease from 13 patient attributes (age, blood pressure, cholesterol, chest pain type, and others), evaluated with a focus on **recall** — since missing a true positive case carries a higher real-world cost than a false alarm in a medical screening context.

## Dataset

- **Source**: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease) (Cleveland subset)
- **Size**: 303 patients, 13 features, reduced to 297 after removing 6 rows with missing values in `ca`/`thal`
- **Target**: Binarized from the original 0–4 severity scale to 0 (no disease) / 1 (disease present)
- **Class balance**: ~54% no disease, ~46% disease — reasonably balanced

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (1–4) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy (0–3) |
| `thal` | Thalassemia type |
| `target` | 0 = no disease, 1 = disease present |

## Approach

1. **Data cleaning** — dropped 6 rows (~2%) with missing `ca`/`thal` values rather than imputing, since these are the strongest predictors and imputation risked distorting the signal the models rely on most.
2. **Preprocessing** — one-hot encoded nominal categorical features (`cp`, `restecg`, `slope`, `thal`); standard-scaled continuous numeric features; left binary features untouched. Fit exclusively on the training split to prevent data leakage.
3. **Train/test split** — 80/20, stratified on the target to preserve class balance in both sets.
4. **Modeling** — trained and compared three classifiers: Logistic Regression (baseline), Random Forest, and XGBoost.
5. **Evaluation** — used 5-fold cross-validation (scored on recall) on the training set for honest, low-variance model comparison, reserving the test set for a single final check.
6. **Hyperparameter tuning** — grid search on Random Forest and XGBoost, focused on reducing overfitting given the small dataset size (~237 training rows).
7. **Feature importance** — extracted from the final XGBoost model and cross-checked against exploratory correlation analysis.

## Model Comparison

| Model | CV Mean Recall | Test Recall | Test Accuracy | Test F1 |
|---|---|---|---|---|
| Logistic Regression | 0.789 | 0.857 | 0.90 | 0.889 |
| Random Forest (tuned) | 0.752 | — | — | — |
| **XGBoost (tuned)** | **0.807** | 0.821 | 0.817 | 0.807 |

**Best params (XGBoost)**: `max_depth=2`, `learning_rate=0.2`, `n_estimators=100`

XGBoost achieved the best cross-validated recall and was selected as the final model. Performance across all three models is closely clustered — a realistic outcome given the dataset's small size (297 patients), and one worth reporting honestly rather than overstating a single "winning" model.

## Feature Importance

Top predictors identified by the final XGBoost model:

1. `cp_4` (chest pain type 4)
2. `thal_7.0` (reversible defect)
3. `ca` (number of major vessels)
4. `sex`
5. `slope_2`
6. `oldpeak`
7. `thalach`

These largely align with the strongest correlations found during exploratory data analysis (`thal`, `ca`, `cp`, `exang`, `oldpeak`), with the model adding precision by identifying which specific categories within `cp` and `thal` matter most.

## Project Structure

```
heart-disease-risk-prediction/
├── data/
│   ├── raw/                    # original dataset (gitignored)
│   └── processed/              # cleaned train/test splits (gitignored)
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
│   ├── download_data.py        # fetches dataset from UCI repo
│   └── predict.py              # loads saved model, predicts on new patient data
├── models/
│   ├── preprocessor.joblib     # fitted encoder/scaler (gitignored)
│   └── heart_disease_xgb_model.joblib  # trained XGBoost model (gitignored)
├── requirements.txt
└── README.md
```

## How to Run

```bash
# 1. Clone and set up environment
git clone <repo-url>
cd heart-disease-risk-prediction
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Download the dataset
python src/download_data.py

# 3. Run the notebook for full exploration and training
jupyter notebook notebooks/01_data_exploration.ipynb

# 4. Predict on a new patient
python src/predict.py
```

## Key Learnings

- Simpler models (Logistic Regression) can be highly competitive with — and sometimes outperform — more complex ensemble methods on small tabular datasets; model complexity should be justified by data volume, not assumed to be better by default.
- Cross-validation is essential for honest model comparison; single test-set metrics on small samples (~60 patients here) can be misleading due to variance.
- Hyperparameter tuning aimed at reducing overfitting (shallower trees, fewer estimators) meaningfully improved both Random Forest and XGBoost.
- Feature importance from the final model was consistent with exploratory correlation analysis, reinforcing confidence in the result.

## Disclaimer

This project is for educational and portfolio purposes only. It is not a validated clinical tool and should not be used for actual medical diagnosis or decision-making.
