# Philippine Housing Price Estimator

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A machine learning project that predicts Philippine house prices using property characteristics and location data. Built with Python, scikit-learn, XGBoost, and Streamlit.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering) and a short `-` delimited description,
│                         e.g. `1.0-initial-data-exploration`.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── house_price_estimator   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes house_price_estimator a Python module
    │
    ├── predict.py              <- Code to run model inference with trained models
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   └── pipeline.py         <- Code for compiling the ML model pipeline
    │   └── train.py            <- Code to train the model
    │
    ├── preprocessing                
    │   ├── __init__.py 
    │   └── cleaning.py         <- Code for deterministic cleaning of data
    │   └── feature_engineering.py  <- Code for deterministic feature engineering
    │   └── transformer.py      <- Code for data processing that will be included in the pipeline
    │
```

--------

## Setup

Create and activate a virtual environment.

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

**1. Train the model**

Run this from the project root directory:

```bash
python3 -m house_price_estimator.modeling.train
```

This trains the model and saves the fitted pipeline to the `models/` directory.

**2. Launch the Streamlit app**

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to use the house price estimator.

## Notes

- Run the training script before launching the app if no trained model exists.
- The Streamlit app uses the saved model pipeline for predictions.
- Notebooks contain exploratory data analysis, feature engineering, and model experiments.
