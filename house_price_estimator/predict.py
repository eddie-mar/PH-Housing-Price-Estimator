import joblib
import os

from house_price_estimator.preprocessing.feature_engineering import det_feature_eng

ROOT = os.getcwd()
MODEL_PATH = os.path.join(ROOT, 'models/', 'xgboost_pipeline.pkl')

def load_model():
    return joblib.load(MODEL_PATH)

def predict(model, data):
    data = det_feature_eng(data)
    return model.predict(data)