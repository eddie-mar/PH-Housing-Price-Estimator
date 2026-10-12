import streamlit as st
import pandas as pd

from house_price_estimator.predict import load_model, predict

model = load_model()

st.title('Philippine House Price Estimator')

description = st.text_input('Property description')
location = st.text_input('Location')
floor_area = st.number_input('Floor Area (sqm)', min_value=1.0)
land_area = st.number_input('Land Area (sqm)', min_value=1.0)
bedrooms = st.number_input('Bedrooms', min_value=1)
bathrooms = st.number_input('Bathrooms', min_value=1)
latitude = st.number_input('Latitude', min_value=-90.0, max_value=90.0)
longitude = st.number_input('Longitude', min_value=-180.0, max_value=180.0)

if st.button('Estimate Price'):
    user_input = pd.DataFrame([{
        'description': description,
        'location': location,
        'floor_area': floor_area,
        'land_area': land_area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'latitude': latitude,
        'longitude': longitude
    }])

    prediction = predict(model, user_input)

    st.success(f'Estimated Price: Php. {prediction[0]:,.0f}')

