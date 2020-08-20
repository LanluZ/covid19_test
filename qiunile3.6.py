import numpy as np
import pandas as pd
import pystan
from fbprophet import Prophet
import matplotlib.pyplot as plt
#导入数据
covid19_data = pd.read_csv("covid19_200820.csv")
covid19_data['ds'] = pd.to_datetime(covid19_data['ds'],format='%Y%m%d')

#调整模型
m = Prophet(changepoint_prior_scale=0.1,changepoints=['2020-05-29','2020-07-20','2020-07-21'])
#传入数据
m.fit(covid19_data)
#预测
future = m.make_future_dataframe(periods=10)
#返回预测值
forecast = m.predict(future)
#画图
m.plot(forecast)
plt.show()
#输出
test = forecast[['ds','yhat']].tail(10)
print(forecast[-10:])