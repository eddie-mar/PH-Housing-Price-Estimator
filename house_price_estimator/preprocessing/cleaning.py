import numpy as np
import pandas as pd

def det_cleaning(data):
    data = data.rename(columns={
        'Description': 'description',
        'Location': 'location',
        'Price': 'price',
        'Bedrooms': 'bedrooms',
        'Bathrooms': 'bathrooms',
        'Floor Area': 'floor_area',
        'Land Area': 'land_area',
        'Latitude': 'latitude',
        'Longitude': 'longitude'
    })

    data = data.drop(columns='HouseID')
    data = data.dropna(subset=['price'])
    data.loc[data['location'].isna(), 'location'] = 'Likha Residences, Muntinlupa'
    data.loc[data['bathrooms'] == 43, 'bathrooms'] = 4

    return data