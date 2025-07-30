import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. Load Dataset
df = pd.read_csv('employee_salary_data.csv')
print('First 5 rows:')
print(df.head())

# 2. EDA
print('\nData Info:')
df.info()
print('\nDescription:')
print(df.describe())

# 3. Preprocessing & Feature Engineering
X = df.drop('Salary', axis=1)
y = df['Salary']
categorical_features = ['Education Level', 'Job Role', 'Location', 'Industry']
numeric_features = ['Experience (years)']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ]
)

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])
rf_pipeline.fit(X_train, y_train)

# 6. Evaluation
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return mae, rmse, r2

print('\nLinear Regression:')
lr_mae, lr_rmse, lr_r2 = evaluate(lr_pipeline, X_test, y_test)
print(f'MAE: {lr_mae:.2f}, RMSE: {lr_rmse:.2f}, R2: {lr_r2:.4f}')

print('\nRandom Forest:')
rf_mae, rf_rmse, rf_r2 = evaluate(rf_pipeline, X_test, y_test)
print(f'MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}, R2: {rf_r2:.4f}')

# 7. Save the Best Model
if rf_r2 > lr_r2:
    joblib.dump(rf_pipeline, 'salary_predictor.joblib')
    print('\nRandom Forest model saved as salary_predictor.joblib')
else:
    joblib.dump(lr_pipeline, 'salary_predictor.joblib')
    print('\nLinear Regression model saved as salary_predictor.joblib')