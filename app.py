import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# print(os.listdir('.'))
df = pd.read_csv('water_consumption_weather_1000.csv')
print(df.head())
df.isnull().sum()
df['Date'] = pd.to_datetime(df['Date'])
df['DayOfWeek'] = df['Date'].dt.dayofweek
df = df[['Temperature_C', 'Humidity_%', 'Rain_mm', 'Wind_kmh', 'Previous_Day_Consumption',
         'Week_Average', 'Month_Average', 'Water_Consumption_Liters', 'DayOfWeek']]
# print( df.columns.tolist())
predict = 'Water_Consumption_Liters'

# فصل المدخلات عن المخرجات و تقسيم البيانات لتدريب واختبار
x = df.drop(columns=[predict])
y = df[predict]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)

# تدريب النموذج
lr = LinearRegression()
lr.fit(x_train, y_train)
c = lr.intercept_
print(c)
m = lr.coef_
print(m)
# التنبؤ على بيانات التدريب و تمثيل التنبؤ
plt.figure()
y_prid_train = lr.predict(x_train)
print(y_prid_train)
plt.scatter(y_train, y_prid_train)
plt.xlabel("Actual Water Consumption (Liters)")
plt.ylabel("Predicted Water Consumption (Liters)")
plt.title("Actual vs Predicted Water Consumption")
plt.show()
# تقييم أداء النموذج على التدريب
r2_train = r2_score(y_train, y_prid_train)
print(r2_train)
# التنبؤ على بيانات الاختبار و تمثيل التنبؤ
plt.figure()
y_prid_test = lr.predict(x_test)
print(y_prid_test)
plt.scatter(y_test, y_prid_test ,)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Water Consumption (Liters)")
plt.ylabel("Predicted Water Consumption (Liters)")
plt.title("Actual vs Predicted Water Consumption")
plt.show()

# تقييم أداء النموذج على الاختبار
r2_test = r2_score(y_test, y_prid_test)
print(r2_test)
#  حساب مقدار الخطأ (RMSE)
rmse_train = np.sqrt(mean_squared_error(y_train, y_prid_train))
print(rmse_train)
rmse_test = np.sqrt(mean_squared_error(y_test, y_prid_test))
print(rmse_test)
# التنبؤ باستهلاك يوم قادم
new_day = np.array([[25, 60, 0, 12, 420, 415, 418, 2]])
predicted_consumption = lr.predict(new_day)

print(round(predicted_consumption[0], 2), "Liters")
