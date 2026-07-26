from ucimlrepo import fetch_ucirepo

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets

X.to_csv('data/raw/heart_disease_features.csv', index = False)
y.to_csv('data/raw/heart_disease_target.csv', index = False)

print("Data downloaded and saved to data/raw/")