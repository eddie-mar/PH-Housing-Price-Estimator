import numpy as np
import pandas as pd
import os
import joblib

from sklearn.model_selection import GridSearchCV, train_test_split

from house_price_estimator.preprocessing.cleaning import det_cleaning
from house_price_estimator.preprocessing.feature_engineering import det_feature_eng
from house_price_estimator.modeling.pipeline import build_xgboost_pipeline

ROOT = os.getcwd()
DATA_SOURCE = os.path.join(ROOT, 'data/raw/', 'PH_HOUSING.csv')
MODEL_PATH = os.path.join(ROOT, 'models/', 'xgboost_pipeline.pkl')

def main():
    # Load data
    data = pd.read_csv(DATA_SOURCE)

    # Deterministic cleaning and feature engineering
    data = det_cleaning(data)
    data = det_feature_eng(data)

    # Split data
    X = data.copy().drop('price', axis=1)
    y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=73)

    # Modeling
    xgb_pipe = build_xgboost_pipeline()

    param_grid = {
        "model__n_estimators": [200, 400, 500],
        "model__max_depth": [3, 6],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0]
    }

    xgb_grid = GridSearchCV(xgb_pipe, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=1)
    xgb_grid.fit(X_train, y_train)

    # Save best pipeline
    joblib.dump(xgb_grid.best_estimator_, MODEL_PATH)

    # Some metrics
    print('Best Parameters:')
    print(xgb_grid.best_params_)

    print(f'Best CV MAE: {-xgb_grid.best_score_:,.0f}')


if __name__ == '__main__':
    main()