import numpy as np
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = np.load("dataset.npz")
X = data["X"]
y = data["y"]

X_flat = X.reshape(X.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=2222)

model = RegressorChain(RandomForestRegressor(n_estimators=500, max_depth=10, random_state=2222))
model.fit(X_train, y_train)

joblib.dump(model, "multioutput_rf_model.pkl")
print("Model saved successfully!")

model = joblib.load("multioutput_rf_model.pkl")
print("Model loaded successfully!")

y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))

print("predict: ",model.predict(X_train[50].reshape(1, -1)))
print("true: ",y_train[50])
