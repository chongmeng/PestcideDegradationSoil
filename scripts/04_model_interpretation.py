import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
import shap
import pickle
import os
import matplotlib

with open(r'results/xgboost_data.pkl', 'rb') as f:
    X, y = pickle.load(f)

# Log-transform specific features
X_log_transformed = X.copy()

# Initialize the model
model = xgb.XGBRegressor()
# Load the model from the JSON file
model.load_model(r'results/xgboost_model.json')

# Define feature names
feature_names = X.columns if hasattr(X, 'columns') else [f'Feature {i}' for i in range(X.shape[1])]

# Plot PDP for all features, with 4 columns
fig, ax = plt.subplots(figsize=(20, 28))
PartialDependenceDisplay.from_estimator(
    model, X_log_transformed, features=range(X.shape[1]), 
    feature_names=feature_names, 
    n_cols=5,  # Change number of columns to 4
    ax=ax
)

plt.suptitle('Partial Dependence Plots')
plt.show()

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_log_transformed)

# SHAP Summary Plot
plt.figure(figsize=(24, 16))
shap.summary_plot(shap_values, X_log_transformed, feature_names=feature_names,show=False)
plt.suptitle('SHAP Summary Plot')
plt.show()
