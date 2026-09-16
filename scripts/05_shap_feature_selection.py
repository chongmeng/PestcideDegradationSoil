import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import mean_squared_error,r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import shap


rs =42 
#use the best_params from 02_hyperparameter_optimization.py
learning_rate=0.1
n_estimators=200
max_depth=7
min_child_weight=5
gamma=0
subsample=0.9
reg_alpha = 1
reg_lambda = 1
df = pd.read_excel("data/data.xlsx", engine='openpyxl')

df = df.drop(df.columns[[0]], axis=1)
df = df[df['DT50'] < 1000]

label_encoder = LabelEncoder()
for column in df.columns:
    # Check if the column contains non-numeric data
    if df[column].dtype == 'object':  # Assuming non-numeric data is of type 'object'
        # Apply label encoding to the column
        df[column] = label_encoder.fit_transform(df[column])

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# Separate features and target variable
X = df.drop(columns='DT50')
y = np.log1p(df['DT50'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)

scaler=MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define cross-validation strategy
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=rs)
# Initialize and train the XGBoost regressor
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,min_child_weight=min_child_weight, 
                          gamma=gamma,subsample=subsample,reg_alpha = reg_alpha,reg_lambda = reg_lambda,random_state=rs)
#xg_reg = xgb.XGBRegressor(objective='reg:squarederror', **best_params,random_state=42)
xg_reg.fit(X_train, y_train)


#Calculate SHAP values on training set
explainer = shap.Explainer(xg_reg)
shap_values = explainer(X_train)

# Calculate mean absolute SHAP value per feature
shap_importance = np.abs(shap_values.values).mean(axis=0)
feature_names = X.columns
shap_summary = pd.DataFrame({
    'feature': feature_names,
    'importance': shap_importance
}).sort_values(by='importance', ascending=False)

results = []
for i in range(1, len(shap_summary) + 1):
    selected_features = shap_summary['feature'].iloc[:i].values
    feature_indices = [X.columns.get_loc(feat) for feat in selected_features]
    
    select_X_all = X_train[:, feature_indices]
    #select_X_test = X_test[:, feature_indices]
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=rs
    )
    
    cv_scores = cross_val_score(model, select_X_all, y_train, cv=cv_strategy, scoring='r2')###
    cv_r2_mean = cv_scores.mean()
    cv_mse_scores = cross_val_score(model, select_X_all, y_train, cv=cv_strategy, scoring='neg_mean_squared_error')
    cv_mse_mean = -cv_mse_scores.mean()
    results.append({
        'num_features': i,
        'cv_r2_mean': cv_r2_mean,
        'cv_mse': cv_mse_mean
    })

results_df = pd.DataFrame(results)
print("\nSummary of all feature subsets:")
print(results_df)
