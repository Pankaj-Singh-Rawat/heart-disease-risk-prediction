# Heart Disease Risk Prediction

A machine learning classifier that predicts heart disease risk from patient demographic and clinical attributes, using the UCI Heart Disease (Cleveland) dataset.

## Problem

Early identification of heart disease risk from routine clinical measurements can support timely intervention. This project builds and compares multiple classification models to predict the presence of heart disease from 13 patient attributes (age, blood pressure, cholesterol, chest pain type, and others), evaluated with a focus on **recall** — since missing a true positive case carries a higher real-world cost than a false alarm in a medical screening context.

## Dataset

- **Source**: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease) (Cleveland subset)
- **Size**: 303 patients, 13 features, reduced to 297 after removing 6 rows with missing values in `ca`/`thal`
- **Target**: Binarized from original 0–4 severity scale to 0 (no disease) / 1 (disease present)
- **Class balance**: ~54% no disease, ~46% disease — reasonably balanced