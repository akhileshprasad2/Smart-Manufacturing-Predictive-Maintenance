
# Big Data Analytics for Predictive Maintenance in Smart Manufacturing Systems

## Project Overview

This MCA project uses machine learning and big data analytics
to predict machine failures in smart manufacturing systems.

## Datasets

1. AI4I 2020 Predictive Maintenance Dataset
2. SECOM Semiconductor Manufacturing Dataset

## Machine Learning Models

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- XGBoost

Best selected models:
- AI4I: XGBoost
- SECOM: Logistic Regression

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- HTML
- CSS
- JavaScript
- Google Colab

## Features

- AI4I manual machine failure prediction
- SECOM CSV batch prediction
- Model comparison
- Prediction history
- Machine health recommendations
- CSV prediction download
- Interactive web dashboard

## Run Locally

Install dependencies:

    pip install -r requirements.txt

Start the application:

    python app.py

Open:

    http://127.0.0.1:5000

## Project Structure

    Smart_Manufacturing_Flask/
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── project_metadata.json
    ├── models/
    ├── data/
    ├── results/
    ├── templates/
    ├── static/
    │   ├── css/
    │   └── js/
    └── uploads/
