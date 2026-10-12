import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from house_price_estimator.preprocessing.transformer import FinalProcessor, KMeansCluster

categorical_features = ['house_type']

def build_xgboost_pipeline():
    target_encoder = TargetEncoder(cols=['city'], handle_missing='value', handle_unknown='value')
    preprocessor = ColumnTransformer(transformers=[('house_type', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)], remainder='passthrough')

    xgb_model = XGBRegressor(random_state=73, n_jobs=-1)

    xgb_pipe = Pipeline([
        ('final_process', FinalProcessor()),
        ('kmeans', KMeansCluster()),
        ('target_encoder', target_encoder),
        ('encoder', preprocessor),
        ('model', xgb_model)
    ])

    return xgb_pipe
