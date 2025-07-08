import joblib

preprocessor = joblib.load('copd_models/preprocessor-scaler.pkl')
scaler = preprocessor.named_transformers_['StandardScaling']

means = scaler.mean_
scales = scaler.scale_

print("Means:", means)
print("Scales:", scales)