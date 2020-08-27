import numpy as np
import pandas as pd
import pystan
from fbprophet import Prophet
import matplotlib.pyplot as plt
#导入数据
covid19_data = pd.read_csv("covid19_200825.csv")
covid19_data['ds'] = pd.to_datetime(covid19_data['ds'],format='%Y%m%d')

#调整模型
m = Prophet(growth='logistic',changepoint_prior_scale=2)
covid19_data['cap']=40000000
m.fit(covid19_data)
#预测
future = m.make_future_dataframe(periods=10)
future['cap']=40000000
#返回预测值
forecast = m.predict(future)

#输出
test = forecast[['ds','yhat']].tail(10)
print(forecast[-10:])

m.plot(forecast)
plt.show()