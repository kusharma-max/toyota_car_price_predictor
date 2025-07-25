# model.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.core.algorithms import duplicated
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load data
df = pd.read_csv("ToyotaCorolla.csv")
df.columns=df.columns.str.lower()
df = df.rename(columns={'age_08_04': 'age_months'})
print(df.columns)

##get unique values
for col in ['doors', 'cylinders', 'gears', 'fuel_type']:
    print(f"Unique values in '{col}':", df[col].unique())

 #get min, max
for col in ['price', 'age_months', 'km', 'hp', 'automatic', 'cc',
       'doors', 'cylinders', 'gears', 'weight']:
    print(f"min in {col} :", df[col].min())
    print(f"max in {col} :", df[col].max())

df.columns = df.columns.str.strip().str.lower()  # normalize headers

df['price'] = df['price'] * 90

df=pd.get_dummies(df,columns=['fuel_type'],drop_first=True)
scaler=StandardScaler()
features=['age_months', 'km', 'hp', 'automatic', 'cc','doors', 'cylinders', 'gears', 'weight']
df[features]=pd.DataFrame(scaler.fit_transform(df[features]))

#check na,duplicates, then remove
print(df.isna().sum())
print(df.duplicated().sum())
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(df.duplicated().sum())
print(df.columns)

X=df.drop('price', axis=1)
y=df['price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=30)
param_grid = {
  'n_estimators': [50, 100, 200],
  'max_depth': [None, 10, 20],
}
grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# model.fit(X_train,y_train)
y_pred = best_model.predict(X_test)
print("R² score:      ", r2_score(y_test, y_pred))
print("MAE:            ", mean_absolute_error(y_test, y_pred))
print("RMSE:           ", np.sqrt(mean_squared_error(y_test, y_pred)))

# ----- Save artifacts -----
joblib.dump(model, "toyota_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")
print("✅ Model, scaler, and column list saved.")