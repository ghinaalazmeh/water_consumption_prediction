import joblib
from sklearn.linear_model import LinearRegression
import numpy as np
model =joblib.load('water_Consumption_model.pkl')
print(type(model))
new_day = np.array([[25, 60, 0, 12, 420, 415, 418, 2]])
predicted_consumption = model.predict(new_day)

print(round(predicted_consumption[0], 2), "Liters")