from data_management import data_manage
import seaborn as sns
import matplotlib.pyplot as plt
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
Analyse the relationship between the number of hospital beds available for allocation in each country, 
i.e. the number of beds per 1,000 population, and the total number of deaths in each country
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

# 绘制散点图
plt.figure(figsize=(10, 6))
sns.regplot(x=x, y=y, data=latest_data, fit_reg=True)
plt.xlabel('Hospital Beds per Thousand')
plt.ylabel('Total Deaths')
plt.title('Relationship between Hospital Beds per Thousand and Total Deaths (Latest Date)')
plt.show()


'''
Analyse the relationship between the number of hospital beds available for allocation in each country, 
i.e. the number of beds per 1,000 population, and the total number of deaths per thousand in each country
'''

# 构建线性回归模型
model = smf.ols(formula='total_deaths ~ hospital_beds_per_thousand', data=latest_data)

# 拟合模型
results = model.fit()

# 打印模型摘要
print("\n Hospital Beds per Thousand and Total Deaths per Thousand")
print(results.summary())

y1 = latest_data['mortality_rate'].values.reshape(-1, 1)

# 绘制散点图
plt.figure(figsize=(10, 6))
sns.regplot(x=x, y=y1, data=latest_data, fit_reg=True)
plt.xlabel('Hospital Beds per Thousand')
plt.ylabel('Total Deaths per Thousand')
plt.title('Relationship between Hospital Beds per Thousand and Total Deaths per Thousand (Latest Date)')
plt.show()


'''
Analysing the relationship between the population of each country and the total number of deaths in each country
'''

# 构建线性回归模型
model = smf.ols(formula='total_deaths ~ population', data=latest_data)

# 拟合模型
results = model.fit()

# 打印模型摘要
print("\n Population and Total Deaths")
print(results.summary())

x1 = latest_data['population'].values.reshape(-1, 1)

# 绘制散点图
plt.figure(figsize=(10, 6))
sns.regplot(x=x1, y=y, data=latest_data, fit_reg=True)
plt.xlabel('Population')
plt.ylabel('Total Deaths')
plt.title('Relationship between Population and Total Deaths (Latest Date)')
plt.show()


'''
Analysing the relationship between the population of each country and the total number of deaths per country in thousands of persons
'''

# 构建线性回归模型
model = smf.ols(formula='total_deaths ~ population', data=latest_data)

# 拟合模型
results = model.fit()

# 打印模型摘要
print("\n Population and Total Deaths per Thousand")
print(results.summary())

# 绘制散点图
plt.figure(figsize=(10, 6))
sns.regplot(x=x1, y=y1, data=latest_data, fit_reg=True)
plt.xlabel('Population')
plt.ylabel('Total Deaths per Thousand')
plt.title('Relationship between Population and Total Deaths per Thousand (Latest Date)')
plt.show()
