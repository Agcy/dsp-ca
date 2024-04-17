from data_management import data_manage
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

# 加载数据集
datapath = '../dataset/clean_data/clean_merged_data.csv'
savepath = '../outputImg/hospital_data/'
data = data_manage(datapath)
data1 = data.copy()

subset1 = data1[data1['indicator'] == 'Daily ICU occupancy']
subset2 = data1[data1['indicator'] == 'Daily ICU occupancy per million']
subset3 = data1[data1['indicator'] == 'Daily hospital occupancy']
subset4 = data1[data1['indicator'] == 'Daily hospital occupancy per million']

'''
分析每个国家医院ICU的占用量与每个国家的当日新增死亡人数之间的关系
'''

# 构建线性回归模型
model = smf.ols(formula='new_deaths ~ value', data=subset1)

# 拟合模型
results = model.fit()

# 打印模型摘要
print("\n Daily ICU occupancy and New Deaths Number")
print(results.summary())

# 提取解释变量和响应变量
x = subset1['value'].values.reshape(-1, 1)
y = subset1['new_deaths'].values.reshape(-1, 1)

# 拟合线性回归模型
regression_model = LinearRegression()
regression_model.fit(x, y)

# 预测响应变量
y_pred = regression_model.predict(x)

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(x, y)
plt.plot(x, y_pred, color='red', linewidth=2)
plt.xlabel('Daily ICU occupancy')
plt.ylabel('New Deaths Number')
plt.xticks(rotation=45)
plt.title('Relationship between Daily ICU occupancy and New Deaths Number')
plt.show()

'''
分析每个国家医院ICU的每一百万人占用量与每个国家的当日新增死亡人数之间的关系
'''

# 构建线性回归模型
model = smf.ols(formula='new_deaths ~ value', data=subset2)

# 拟合模型
results = model.fit()

# 打印模型摘要
print("\n Daily ICU occupancy per million people and New Deaths Number")
print(results.summary())

# 提取解释变量和响应变量
x1 = subset2['value'].values.reshape(-1, 1)
y1 = subset2['new_deaths'].values.reshape(-1, 1)

# 拟合线性回归模型
regression_model = LinearRegression()
regression_model.fit(x1, y1)

# 预测响应变量
y1_pred = regression_model.predict(x1)

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(x1, y1)
plt.plot(x1, y1_pred, color='red', linewidth=2)
plt.xlabel('Daily ICU occupancy per million people')
plt.ylabel('New Deaths Number')
plt.xticks(rotation=45)
plt.title('Relationship between Daily ICU occupancy per million people and New Deaths Number')
plt.show()

'''
分析每个国家医院的占用量与每个国家的当日新增死亡人数之间的关系
'''

# 构建线性回归模型
model = smf.ols(formula='new_deaths ~ value', data=subset3)

# 拟合模型
results = model.fit()

# 打印模型摘要
print("\n Daily Hospital occupancy and New Deaths Number")
print(results.summary())

# 提取解释变量和响应变量
x2 = subset3['value'].values.reshape(-1, 1)
y2 = subset3['new_deaths'].values.reshape(-1, 1)

# 拟合线性回归模型
regression_model = LinearRegression()
regression_model.fit(x2, y2)

# 预测响应变量
y2_pred = regression_model.predict(x2)

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(x2, y2)
plt.plot(x2, y2_pred, color='red', linewidth=2)
plt.xlabel('Daily Hospital occupancy')
plt.ylabel('New Deaths Number')
plt.xticks(rotation=45)
plt.title('Relationship between Daily Hospital occupancy and New Deaths Number')
plt.show()

'''
分析每个国家医院的每一百万人占用量与每个国家的当日新增死亡人数之间的关系
'''

# 构建线性回归模型
model = smf.ols(formula='new_deaths ~ value', data=subset4)

# 拟合模型
results = model.fit()

# 打印模型摘要
print("\n Daily Hospital occupancy per million people and New Deaths Number")
print(results.summary())

# 提取解释变量和响应变量
x3 = subset4['value'].values.reshape(-1, 1)
y3 = subset4['new_deaths'].values.reshape(-1, 1)

# 拟合线性回归模型
regression_model = LinearRegression()
regression_model.fit(x3, y3)

# 预测响应变量
y3_pred = regression_model.predict(x3)

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(x3, y3)
plt.plot(x3, y3_pred, color='red', linewidth=2)
plt.xlabel('Daily Hospital occupancy per million people')
plt.ylabel('New Deaths Number')
plt.xticks(rotation=45)
plt.title('Relationship between Daily Hospital occupancy per million people and New Deaths Number')
plt.show()

'''
分析每个国家医院ICU的病人人数与每个国家的当日新增死亡人数之间的关系
'''

# 构建线性回归模型
model = smf.ols(formula='new_deaths ~ icu_patients', data=subset2)

# 拟合模型
results = model.fit()

# 打印模型摘要
print("\n ICU Patients and New Deaths Number")
print(results.summary())

# 提取解释变量和响应变量
x4 = subset2['icu_patients'].values.reshape(-1, 1)
y4 = subset2['new_deaths'].values.reshape(-1, 1)

# 拟合线性回归模型
regression_model = LinearRegression()
regression_model.fit(x4, y4)

# 预测响应变量
y4_pred = regression_model.predict(x4)

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(x4, y4)
plt.plot(x4, y4_pred, color='red', linewidth=2)
plt.xlabel('ICU Patients Number')
plt.ylabel('New Deaths Number')
plt.xticks(rotation=45)
plt.title('Relationship between ICU Patients Number and New Deaths Number')
plt.show()

'''
分析每个国家医院病人的人数与每个国家的当日新增死亡人数之间的关系
'''

# 构建线性回归模型
model = smf.ols(formula='new_deaths ~ hosp_patients', data=subset2)

# 拟合模型
results = model.fit()

# 打印模型摘要
print("\n Hospital Patients and New Deaths Number")
print(results.summary())

# 提取解释变量和响应变量
x5 = subset2['hosp_patients'].values.reshape(-1, 1)
y5 = subset2['new_deaths'].values.reshape(-1, 1)

# 拟合线性回归模型
regression_model = LinearRegression()
regression_model.fit(x5, y5)

# 预测响应变量
y5_pred = regression_model.predict(x5)

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(x5, y5)
plt.plot(x5, y5_pred, color='red', linewidth=2)
plt.xlabel('Patients Number in Hospital')
plt.ylabel('New Deaths Number')
plt.xticks(rotation=45)
plt.title('Relationship between Patients Number in Hospital and New Deaths Number')
plt.show()
