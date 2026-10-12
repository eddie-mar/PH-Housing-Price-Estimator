import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin

# Final Processor
class FinalProcessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = X.copy()

        # Floor and Land Area
        self.floor_to_land_ratio = (X['floor_area'] / X['land_area']).median()
        self.floor_median = X['floor_area'].median()
        self.land_median = X['land_area'].median()

        floor_area_for_bins = X['floor_area'].fillna(X['land_area'] * self.floor_to_land_ratio)
        floor_area_for_bins = floor_area_for_bins.fillna(self.floor_median)
        
        # Bathroom and Bedroom
        _, self.bin_edges = pd.qcut(floor_area_for_bins, q=10, retbins=True, duplicates='drop')
        floor_bins = pd.cut(floor_area_for_bins, bins=self.bin_edges)

        self.bedroom_medians = X.groupby(floor_bins, observed=True)['bedrooms'].median()
        self.bathroom_medians = X.groupby(floor_bins, observed=True)['bathrooms'].median()

        self.bedroom_median = X['bedrooms'].median()
        self.bathroom_median = X['bathrooms'].median()

        return self

    def transform(self, X):
        X = X.copy()

        # Floor and Land Area
        X['floor_area'] = X['floor_area'].fillna(X['land_area'] * self.floor_to_land_ratio)
        X['land_area'] = X['land_area'].fillna(X['floor_area'] / self.floor_to_land_ratio)

        X['floor_area'] = X['floor_area'].fillna(self.floor_median)
        X['land_area'] = X['land_area'].fillna(self.land_median)
        
        # Bathroom and Bedroom
        X['floor_bin'] = pd.cut(X['floor_area'], bins=self.bin_edges)
        
        X['bedrooms'] = X['bedrooms'].fillna(X['floor_bin'].map(self.bedroom_medians))
        X['bathrooms'] = X['bathrooms'].fillna(X['floor_bin'].map(self.bathroom_medians))

        X['bedrooms'] = X['bedrooms'].fillna(self.bedroom_median)
        X['bathrooms'] = X['bathrooms'].fillna(self.bathroom_median)

        X = X.drop(columns='floor_bin')

        return X


# KMeans

class KMeansCluster(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.city_coords = X.groupby('city')[['latitude', 'longitude']].mean()

        X = X[['latitude', 'longitude']].copy()

        self.model = KMeans(n_clusters=11, random_state=73, n_init='auto')
        self.model.fit(X)

        return self

    def transform(self, X):
        X = X.copy()

        # Fill missing coordinates using city coordinates
        for coord in ['latitude', 'longitude']:
            X[coord] = X[coord].fillna(X['city'].map(self.city_coords[coord]))

        clusters = self.model.predict(X[['latitude', 'longitude']])
        X['cluster'] = clusters

        return X
