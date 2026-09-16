import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import mean_squared_error,r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle

rs = 42

#use the best_params from 02_hyperparameter_optimization.py
learning_rate=0.1
n_estimators=200
max_depth=7
min_child_weight=5
gamma=0
subsample=0.9
reg_alpha = 1
reg_lambda = 1

df = pd.read_excel('data/data.xlsx')
df = df.drop(df.columns[[0]], axis=1)
# Filter rows where DT50 < 1000
df = df[df['DT50'] < 1000]

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# Initialize LabelEncoders and store them
label_encoders = {}  
for column in df.columns:
    if df[column].dtype == 'object':  # If column is categorical
        df[column] = df[column].astype(str)
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])  # Transform and save mapping
        label_encoders[column] = le  

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
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                          min_child_weight=min_child_weight, gamma=gamma,subsample=subsample,reg_alpha = reg_alpha,reg_lambda = reg_lambda,random_state=rs)
xg_reg.fit(X_train, y_train)

# Predict and evaluate the model
train_preds = xg_reg.predict(X_train)
test_preds = xg_reg.predict(X_test)

train_mse = mean_squared_error(y_train, train_preds)
print(f"Training Mean Squared Error: {train_mse}")
test_mse = mean_squared_error(y_test, test_preds)
print(f"Test Mean Squared Error: {test_mse}")

train_r2 = r2_score(y_train, train_preds)
print(f"Training R-squared: {train_r2}")
test_r2 = r2_score(y_test, test_preds)
print(f"Test R-squared: {test_r2}")

cv_r2 = cross_val_score(xg_reg, X_train, y_train, cv=cv_strategy, scoring='r2').mean()
print(f"Cross-Validation R-squared: {cv_r2}")
cv_scores = cross_val_score(xg_reg, X_train, y_train, cv=cv_strategy, scoring='r2')
print(f"Cross-Validation R-squared scores: {cv_scores}")

cv_mse_scores = -cross_val_score(xg_reg, X_train, y_train, cv=cv_strategy, scoring='neg_mean_squared_error')
print(f" Cross-Validation MSE scores: {cv_mse_scores}")
print(f" Mean Cross-Validation MSE: {cv_mse_scores.mean():.4f}")
print(f" Standard Deviation of CV MSE: {cv_mse_scores.std():.4f}")

# PP-Plot function
def pp_plot(actual, predicted, title):
    sorted_actual = np.sort(actual)
    sorted_predicted = np.sort(predicted)
    plt.figure(figsize=(6,6))
    plt.plot(sorted_actual, sorted_predicted, 'o', label='Predicted vs Actual')
    plt.plot(sorted_actual, sorted_actual, 'r--', label='Ideal Line')
    plt.xlabel("Actual DT50 (log-transformed)")
    plt.ylabel("Predicted DT50 (log-transformed)")
    plt.title(title)
    plt.legend()
    plt.show()

# Scatter plot function
def scatter_plot(actual, predicted, title):
    plt.figure(figsize=(6,6))
    plt.scatter(actual, predicted, alpha=0.6)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--')
    plt.xlabel("Actual DT50 (log-transformed)")
    plt.ylabel("Predicted DT50 (log-transformed)")
    plt.title(title)
    plt.show()

# Combined scatter plot for actual vs predicted DT50 (Train & Test together)
def actual_vs_predicted_combined(y_train, train_preds, y_test, test_preds):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=np.expm1(y_train), y=np.expm1(train_preds), label='Train Data', color='blue', alpha=0.6)
    sns.scatterplot(x=np.expm1(y_test), y=np.expm1(test_preds), label='Test Data', color='orange', alpha=0.6)
    plt.plot([min(np.expm1(y_train).min(), np.expm1(y_test).min()), 
              max(np.expm1(y_train).max(), np.expm1(y_test).max())], 
             [min(np.expm1(y_train).min(), np.expm1(y_test).min()), 
              max(np.expm1(y_train).max(), np.expm1(y_test).max())], 
             'r--', linewidth=2)  # Diagonal line
    plt.xlabel("Actual DT50")
    plt.ylabel("Predicted DT50")
    plt.title("Actual vs. Predicted DT50 (Train & Test)")
    plt.legend()
    plt.show()



# Plot PP-plot and scatter plot for training data
pp_plot(y_train, train_preds, "P-P Plot for Training Data")
scatter_plot(y_train, train_preds, "Predicted vs Actual DT50 - Training Data")

# Plot PP-plot and scatter plot for test data
pp_plot(y_test, test_preds, "P-P Plot for Test Data")
scatter_plot(y_test, test_preds, "Predicted vs Actual DT50 - Test Data")

# Plot combined actual vs predicted DT50
actual_vs_predicted_combined(y_train, train_preds, y_test, test_preds)


#save the model and data
xg_reg.save_model(r'results/xgboost_model.json')

# Save LabelEncoders and Scaler
with open(r'results/xgboost_scaler.pkl', 'wb') as f:
    pickle.dump({r'label_encoders': label_encoders, 'scaler': scaler}, f)

# Save transformed data (for later comparison if needed)
X_train_scaled_df = pd.DataFrame(X_train, columns=X.columns)
with open(r'results/xgboost_data.pkl', 'wb') as f:
    pickle.dump((X_train_scaled_df, y_train), f)
