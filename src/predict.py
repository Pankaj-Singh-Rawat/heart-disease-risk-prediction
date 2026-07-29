import pandas as pd
import joblib

def load_artifacts():
    """Load the fitted preprocessor and trained model from disk."""
    preprocessor = joblib.load('models/preprocessor.joblib')
    model = joblib.load('models/heart_disease_xgb_model.joblib')
    return preprocessor, model


def predict_risk(patient_data: dict):
    """
    Predict heart disease risk for a single patient.

    patient_data: dict with keys matching the original raw feature columns:
        age, sex, cp, trestbps, chol, fbs, restecg, thalach,
        exang, oldpeak, slope, ca, thal
    """
    preprocessor, model = load_artifacts()

    # Convert single patient dict into a one-row DataFrame
    patient_df = pd.DataFrame([patient_data])

    # Apply the SAME transformations used during training
    patient_processed = preprocessor.transform(patient_df)

    prediction = model.predict(patient_processed)[0]
    probability = model.predict_proba(patient_processed)[0][1]

    return {
        'prediction': 'Disease Likely' if prediction == 1 else 'Low Risk',
        'probability': round(float(probability), 3)
    }


if __name__ == '__main__':
    # Example patients
    sample_patient = {
        'age': 63, 'sex': 1, 'cp': 4, 'trestbps': 145, 'chol': 233,
        'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
        'oldpeak': 2.3, 'slope': 3, 'ca': 0, 'thal': 6
    }

    result = predict_risk(sample_patient)
    print(f"Prediction: {result['prediction']}")
    print(f"Probability of disease: {result['probability']}")