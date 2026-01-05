import numpy as np
import xgboost as xgb
from sklearn.multioutput import RegressorChain
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

data = np.load("dataset.npz")
X = data["X"]
y = data["y"]

X_flat = X.reshape(X.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=2222)

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    subsample=0.6,
    colsample_bytree=0.65,
    objective='reg:squarederror',
    random_state=2222
)

model = RegressorChain(xgb_model)

model.fit(X_train, y_train)

joblib.dump(model, "multioutput_xgb_model.pkl")
print("XGBoost Model saved successfully!")

y_pred = model.predict(X_test)

print("-" * 30)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))
print("-" * 30)

sample_idx = 50
if sample_idx < len(X_train):
    print("Predict:", model.predict(X_train[sample_idx].reshape(1, -1)))
    print("True Value:", y_train[sample_idx])
