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

# 选择需要绘制的变量
variables = ['hospital_beds_per_thousand', 'icu_patients', 'hosp_patients', 'total_deaths', 'new_deaths']

# 按日期降序排序
data_sorted = data1.sort_values('date', ascending=False)

# 获取每个国家的最新数据
latest_data = data_sorted.groupby('entity').first().reset_index()
# # 筛选数据
# selected_data = data1[data1['indicator'].isin(variables)]
#
# # 获取每个国家的最新日期的数据
# latest_date_data = selected_data.groupby('entity').apply(lambda x: x[x['date'] == x['date'].max()])


'''
分析每个国家医院可分配的病床数量，即每一千个人有多少病床与每个国家的总死亡人数之间的关系
'''

# 构建线性回归模型
model = smf.ols(formula='total_deaths ~ hospital_beds_per_thousand', data=latest_data)

# 拟合模型
results = model.fit()

# 打印模型摘要
print("\n Hospital Beds per Thousand and Total Deaths")
print(results.summary())

# 提取解释变量和响应变量
x = latest_data['hospital_beds_per_thousand'].values.reshape(-1, 1)
y = latest_data['total_deaths'].values.reshape(-1, 1)

# 拟合线性回归模型
regression_model = LinearRegression()
regression_model.fit(x, y)

# 预测响应变量
y_pred = regression_model.predict(x)

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(x, y)
plt.plot(x, y_pred, color='red', linewidth=2)
plt.xlabel('Hospital Beds per Thousand')
plt.ylabel('Total Deaths')
plt.title('Relationship between Hospital Beds per Thousand and Total Deaths (Latest Date)')
plt.show()


'''
分析每个国家医院可分配的病床数量，即每一千个人有多少病床与每个国家的总死亡人数千分比之间的关系
'''

# 构建线性回归模型
model = smf.ols(formula='total_deaths ~ hospital_beds_per_thousand', data=latest_data)

# 拟合模型
results = model.fit()

# 打印模型摘要
print("\n Hospital Beds per Thousand and Total Deaths per Thousand")
print(results.summary())

y1 = latest_data['mortality_rate'].values.reshape(-1, 1)
# 拟合线性回归模型
regression_model1 = LinearRegression()
regression_model1.fit(x, y1)

# 预测响应变量
y1_pred = regression_model1.predict(x)
# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(x, y1)
plt.plot(x, y1_pred, color='red', linewidth=2)
plt.xlabel('Hospital Beds per Thousand')
plt.ylabel('Total Deaths per Thousand')
plt.title('Relationship between Hospital Beds per Thousand and Total Deaths per Thousand (Latest Date)')
plt.show()


'''
分析每个国家人口与每个国家的总死亡人数之间的关系
'''

# 构建线性回归模型
model = smf.ols(formula='total_deaths ~ hospital_beds_per_thousand', data=latest_data)

# 拟合模型
results = model.fit()

# 打印模型摘要
print("\n Population and Total Deaths")
print(results.summary())

x1 = latest_data['population'].values.reshape(-1, 1)
# 拟合线性回归模型
regression_model2 = LinearRegression()
regression_model2.fit(x1, y)

# 预测响应变量
y_pred2 = regression_model2.predict(x1)
# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(x1, y)
plt.plot(x1, y_pred2, color='red', linewidth=2)
plt.xlabel('Population')
plt.ylabel('Total Deaths')
plt.title('Relationship between Population and Total Deaths (Latest Date)')
plt.show()


'''
分析每个国家人口与每个国家的总死亡人数千分比之间的关系
'''

# 构建线性回归模型
model = smf.ols(formula='total_deaths ~ hospital_beds_per_thousand', data=latest_data)

# 拟合模型
results = model.fit()

# 打印模型摘要
print("\n Population and Total Deaths per Thousand")
print(results.summary())

# 拟合线性回归模型
regression_model3 = LinearRegression()
regression_model3.fit(x1, y1)

# 预测响应变量
y1_pred2 = regression_model3.predict(x1)
# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(x1, y1)
plt.plot(x1, y1_pred2, color='red', linewidth=2)
plt.xlabel('Population')
plt.ylabel('Total Deaths per Thousand')
plt.title('Relationship between Population and Total Deaths per Thousand (Latest Date)')
plt.show()
