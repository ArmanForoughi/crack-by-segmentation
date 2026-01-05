import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Input, Flatten, Dense, Dropout 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
import random

SEED = 2222
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

data = np.load("dataset.npz")
X = data["X"]
y = data["y"]

X = X.reshape(X.shape[0], 3, 5, 6).transpose(0, 3, 1, 2)  

X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

model = Sequential([
    Input(shape=(6, 3, 5)),
    Conv2D(32, (2,2), activation='relu', padding='same'),
    Dropout(0.3),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(5, activation='sigmoid')
])

model.compile(optimizer="adam", loss='mse', metrics=['mse'])
model.summary()

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=90, restore_best_weights=True)
model.fit(X_train, y_train, epochs=300, batch_size=16, validation_split=0.2, callbacks=[es])

model.save("cnn_model.keras")

from keras.models import load_model

model = load_model("cnn_model.keras")

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))

print("predict: ",model.predict(X_train[50].reshape(1, 6, 3, 5)))
print("true: ",y_train[50])

for i in range(y.shape[1]):
    mae_i = mean_absolute_error(y_test[:, i], y_pred[:, i])
    rmse_i = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
    print(f"Output {i+1}: MAE={mae_i:.4f}, RMSE={rmse_i:.4f}")

from scipy.stats import pearsonr, spearmanr

y_true_flat = y_test.flatten()
y_pred_flat = y_pred.flatten()

pearson_r = pearsonr(y_true_flat, y_pred_flat)[0]
spearman_r = spearmanr(y_true_flat, y_pred_flat)[0]

print("Pearson correlation:", pearson_r)
print("Spearman correlation:", spearman_r)
