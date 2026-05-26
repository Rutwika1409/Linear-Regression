# Simple Linear Regression 

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Import cleaned dataframe
from eda import df

# Train Test Splitting New Cleaned Dataset
X = df.drop(columns=['Item_Outlet_Sales'])
X = pd.get_dummies(X, drop_first=True)
y = df['Item_Outlet_Sales']

feature_columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Model Training
model = LinearRegression()

model.fit(X_train, y_train)

# Model Prediction
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)

print("Mean Absolute Error:", mae)

print("R^2 Score:", r2)

# Save the model 
import os
import joblib

# Create models folder
os.makedirs("models", exist_ok=True)

# Save the model 
joblib.dump(
    model,
    "models/linear_regression_model.pkl"
)

# Save scaler
joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Model Saved Successfully")