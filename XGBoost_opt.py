import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error,r2_score
from tqdm import tqdm
import xgboost as xgb
import matplotlib.pyplot as plt

rs = 42

df = pd.read_excel(r'data.xlsx')


df = df.drop(df.columns[[0]], axis=1)
label_encoder = LabelEncoder()
for column in df.columns:
    # Check if the column contains non-numeric data
    if df[column].dtype == 'object':  # Assuming non-numeric data is of type 'object'
        # Apply label encoding to the column
        df[column] = label_encoder.fit_transform(df[column])

# Filter rows where DT50 < 1000
df = df[df['DT50'] < 1000]

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1]) 



# Separate features and target variable
X = df.drop(columns='DT50')
y = np.log1p(df['DT50'])
features = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)

scaler=MinMaxScaler()
scaler.fit(X)
X_scaler = scaler.transform(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Define cross-validation strategy
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=rs)

# Initialize and train the XGBoost regressor
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=rs)

# # Define hyperparameters grid for grid search
param_grid = {
    'learning_rate': [0.1,0.01,0.2],
    'n_estimators': [200,300,400,500,600,700,800,900,1000],
    'max_depth': [3,4,5,6,7,8,9],#lower, model simpler
    'min_child_weight': [1,5,10,20],
    'gamma': [0.1,0.5,0,1],
    'subsample': [0.5,0.7,0.9],'reg_alpha': [0.1,0.5,0,1],
    'reg_lambda': [0.1,0.5,1,0]
 }

grid_search = GridSearchCV(estimator=xg_reg, param_grid=param_grid, 
                           scoring='r2', cv=cv_strategy, n_jobs=-1)

grid_search.fit(X_train, y_train)
# Get best parameters and best R-squared score
best_params = grid_search.best_params_
print("Best Parameters:", grid_search.best_params_)

best_score = grid_search.best_score_
print("Best Cross-Validation R-squared:", best_score)



# Use best parameters to train XGBoost model
best_xg_reg = xgb.XGBRegressor(objective='reg:squarederror', **best_params,random_state=rs)
best_xg_reg.fit(X_train, y_train)

# Predict and evaluate the model
train_preds = best_xg_reg.predict(X_train)
test_preds = best_xg_reg.predict(X_test)

train_mse = mean_squared_error(y_train, train_preds)
print(f"Training Mean Squared Error: {train_mse}")
test_mse = mean_squared_error(y_test, test_preds)
print(f"Test Mean Squared Error: {test_mse}")

train_r2 = r2_score(y_train, train_preds)
print(f"Training R-squared: {train_r2}")
test_r2 = r2_score(y_test, test_preds)
print(f"Test R-squared: {test_r2}")
cv_r2 = cross_val_score(best_xg_reg, X_scaler, y, cv=cv_strategy, scoring='r2').mean()
print(f"Cross-Validation R-squared: {cv_r2}")
cv_scores = cross_val_score(best_xg_reg, X_scaler, y, cv=cv_strategy, scoring='r2')
print(f"Cross-Validation R-squared scores on entire data: {cv_scores}")