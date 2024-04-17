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
Analysing the relationship between hospital ICU occupancy in each country and the number of new deaths per day in each country
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

# 绘制散点图
plt.figure(figsize=(10, 6))
sns.regplot(x=x, y=y, data=subset1, fit_reg=True)
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

# 绘制散点图
plt.figure(figsize=(10, 6))
sns.regplot(x=x1, y=y1, data=subset2, fit_reg=True)
plt.xlabel('Daily ICU occupancy per million people')
plt.ylabel('New Deaths Number')
plt.xticks(rotation=45)
plt.title('Relationship between Daily ICU occupancy per million people and New Deaths Number')
plt.show()

'''
Analysing the relationship between hospital occupancy in each country and the number of new deaths per day in each country
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

# 绘制散点图
plt.figure(figsize=(10, 6))
sns.regplot(x=x2, y=y2, data=subset3, fit_reg=True)
plt.xlabel('Daily Hospital occupancy')
plt.ylabel('New Deaths Number')
plt.xticks(rotation=45)
plt.title('Relationship between Daily Hospital occupancy and New Deaths Number')
plt.show()

'''
Analysing the relationship between hospital occupancy per million people in each country and the number of new deaths per day in each country
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

# 绘制散点图
plt.figure(figsize=(10, 6))
sns.regplot(x=x3, y=y3, data=subset4, fit_reg=True)
plt.xlabel('Daily Hospital occupancy per million people')
plt.ylabel('New Deaths Number')
plt.xticks(rotation=45)
plt.title('Relationship between Daily Hospital occupancy per million people and New Deaths Number')
plt.show()

'''
Analysing the relationship between the number of patients in hospital ICUs in each country and the number of new deaths per day in each country
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

# 绘制散点图
plt.figure(figsize=(10, 6))
sns.regplot(x=x4, y=y4, data=subset2, fit_reg=True)
plt.xlabel('ICU Patients Number')
plt.ylabel('New Deaths Number')
plt.xticks(rotation=45)
plt.title('Relationship between ICU Patients Number and New Deaths Number')
plt.show()

'''
Analysing the relationship between the number of hospital patients in each country and the number of new deaths per day in each country
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

# 绘制散点图
plt.figure(figsize=(10, 6))
sns.regplot(x=x5, y=y5, data=subset2, fit_reg=True)
plt.xlabel('Patients Number in Hospital')
plt.ylabel('New Deaths Number')
plt.xticks(rotation=45)
plt.title('Relationship between Patients Number in Hospital and New Deaths Number')
plt.show()
