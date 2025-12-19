
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import mean_absolute_error , mean_squared_error
import json
import joblib
#csv استدعاء وقراءة ملف 
df = pd.read_csv('water_consumption_weather_1000.csv')
#تنظيف البيانات

#اضافة عمود DayOfWeek
df["Date"] = pd.to_datetime(df["Date"])
df["DayOfWeek"] = df["Date"].dt.dayofweek
# تبديل آخر عمودين فقط
cols = df.columns.tolist()
cols[-2], cols[-1] = cols[-1], cols[-2]  # swap آخر عمودين
df = df[cols]
print (df.head())
#التأكد من عدم وجود قيم فارغة
test=df.isnull().sum()
print(test)
#قسم البيانات
#featuresتعريف ال
x = df.drop(columns= ['Date','Water_Consumption_Liters'])
#تعريف الهدف
y=df['Water_Consumption_Liters']
#تقسيم مجموعة البيانات
x_train , x_test ,y_train ,  y_test =train_test_split(x , y , test_size=0.3 , random_state=0)
lr=LinearRegression()
#تدريب النموذج
lr.fit(x_train , y_train)
# y نقطة تقاطع محور 
intercept=lr.intercept_
print("intercept=",intercept)
#معاملات كل ميزة (كم يؤثر كل ميزه على التوقع)
coefficients=lr.coef_
feature_names = x.columns
feature_importance = dict(zip(x.columns, lr.coef_))
print("coefficients", {k: round(float(v), 3) for k, v in feature_importance.items()})
#التأكد من أن هذا النموذج يعمل
y_pred_train = lr.predict(x_train)
print(y_pred_train)
#  نمثيل التنبؤ التدريب
plt.scatter(y_train , y_pred_train)
plt.xlabel("Actual Water Consumption Liters")
plt.ylabel("Predicted Water Consumption Liters")
#plt.show()
# جودة التنبؤ
#حساب r2
r2=r2_score(y_train , y_pred_train)
print("R²:train" , r2)
#حساب rmse
rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print("RMSE :train", rmse)

#  نمثيل تنبؤ الاختبار 
y_pred_test = lr.predict(x_test)
plt.scatter(y_test , y_pred_test)
plt.xlabel("Actual Water Consumption Liters")
plt.ylabel("Predicted Water Consumption Liters")
plt.show()
# جودة التنبؤ
#حساب r2
r2=r2_score(y_test , y_pred_test)

#MAE حساب
mae=mean_absolute_error (y_test , y_pred_test)
#MSE حساب
mse=mean_squared_error (y_test , y_pred_test)
#حساب rmse
rmse =np.sqrt(mse)

#MAPE حساب
mape=np.mean(np.abs((y_test - y_pred_test)/ y_pred_test)*100)

#النتائج
print("R²:" , r2)
print("MAE:" ,mae )
print("MSE:" , mse)
print("rmse:" ,rmse )
print("MAPE:" ,mape )
#حفظ النموذج المدرب
joblib.dump(lr , 'water_Consumption_model.pkl')

#حفظ في ملف json
result = {
    "model_version": "linear-regression",
    "r2": float(round(r2, 3)),
    "rmse":float(round(rmse, 2)),
    "feature_importance": {k: float(round(v, 3)) for k, v in feature_importance.items()}
}

with open("model.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4)

print(result)