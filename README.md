# California Wildfire Cause Prediction Using Machine Learning

## Project Overview

This project aims to predict the causes of California wildfires—whether human-induced or natural—using machine learning techniques. By integrating wildfire incident data with meteorological and geospatial information, the model offers actionable insights to help mitigate wildfire risks.

Over 85% of wildfires in California are caused by human activities. Accurately identifying the cause of a wildfire can improve resource allocation and prevention strategies.

---

## Dataset

- **FRAP Wildfire Data:** California Department of Forestry and Fire Protection (CAL FIRE) wildfire incidents dataset.
- **NOAA Weather Data:** Historical meteorological data including temperature, wind speed/direction, and other relevant variables.

---

## Features

- Spatial features based on latitude and longitude
- Temporal cyclical features (hour of day, month of year) encoded as sine and cosine
- Temperature extremes (TMIN, TMAX)
- Fire size
- Wind direction encoded as categorical or cyclical features
- Additional engineered features to improve model accuracy

---

## Data Processing

- Spatial joins using Haversine distance for integrating wildfire and weather data
- Handling missing geospatial data using GeoPandas
- Addressing class imbalance with SMOTE (Synthetic Minority Over-sampling Technique)
- Feature selection with SelectKBest based on mutual information scores

---

## Models

- Logistic Regression
- LightGBM
- XGBoost
- Random Forest (best performing with 94% test accuracy)

---

## Model Deployment

The final Random Forest model is deployed as a Flask API, allowing for real-time predictions of wildfire causes based on input features.

Future Work
Incorporate more granular weather and vegetation data

Experiment with deep learning models for improved accuracy

Enhance API with authentication and scalability features

## Results
Random Forest achieved 94% accuracy on the test set.

Model performs well in identifying minority class (natural wildfire causes) with strong precision and recall.

Feature importance analysis highlights key drivers of wildfire causes.

## Technologies Used
Python

Pandas, NumPy

Scikit-learn

GeoPandas

LightGBM, XGBoost

Flask

Matplotlib, Seaborn

## Sources
FRAP Wildfire Data:
California Department of Forestry and Fire Protection (CAL FIRE) Fire and Resource Assessment Program (FRAP)
https://frap.fire.ca.gov/data/frapgis-data/

NOAA Weather Data:
National Oceanic and Atmospheric Administration (NOAA) National Centers for Environmental Information (NCEI)
https://www.ncdc.noaa.gov/data-access

SMOTE for Imbalanced Data:
Chawla, N. V., et al. "SMOTE: Synthetic Minority Over-sampling Technique." Journal of Artificial Intelligence Research, 2002.

Geospatial Analysis:
GeoPandas Documentation — https://geopandas.org/

## Machine Learning Libraries:
Scikit-learn — https://scikit-learn.org/
LightGBM — https://lightgbm.readthedocs.io/
XGBoost — https://xgboost.readthedocs.io/

## License
This project is licensed under the MIT License.

## Contact
Annissa Pereira
GitHub Profile: https://github.com/annissapereira
LinkedIn Profile: https://www.linkedin.com/in/annissapereira/
Email: annissa.p01@gmail.com


