from data_management import data_manage
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据集
datapath = '../dataset/clean_data/clean_merged_data.csv'
savepath = '../outputImg/hospital_data/'
data = data_manage(datapath)
data1 = data.copy()

# 获取每个国家的每天医院病人数量的第一条数据
hosp_patients_data = data1.groupby(['entity', 'year', 'month', 'day'])['hosp_patients'].first().reset_index()
country_total_patients = hosp_patients_data.groupby('entity')['hosp_patients'].sum()
hosp_patients_data['date'] = pd.to_datetime(hosp_patients_data[['year', 'month', 'day']])
# 获取前十名和后十名数据
top_5_countries = list(country_total_patients.nlargest(5).index)
bottom_5_countries = list(country_total_patients.nsmallest(5).index)
print(top_5_countries)
print(bottom_5_countries)

top_5 = hosp_patients_data[hosp_patients_data['entity'].isin(top_5_countries)]
bottom_5 = hosp_patients_data[hosp_patients_data['entity'].isin(bottom_5_countries)]
# 设置图形大小
plt.figure(figsize=(10, 6))

# 绘制折线图
sns.lineplot(x='date', y='hosp_patients', hue='entity', data=top_5)

# 设置图形标题和轴标签
plt.title('Hospital Patients by Country top 5')
plt.xlabel('Day')
plt.ylabel('Number of Hospital Patients')
plt.xticks(rotation=45)
# 显示图形
plt.savefig(savepath + 'hosp_patients_data_top_5.png')
plt.show()


# 绘制折线图
sns.lineplot(x='date', y='hosp_patients', hue='entity', data=bottom_5)

# 设置图形标题和轴标签
plt.title('Hospital Patients by Country bottom 5')
plt.xlabel('Day')
plt.ylabel('Number of Hospital Patients')
plt.xticks(rotation=45)
plt.savefig(savepath + 'hosp_patients_data_bottom_5.png')
# 显示图形
plt.show()

'''
分析每个国家医院可分配的病床数量，即每一千个人有多少病床
'''
data2 = data.copy()
hosp_beds_per_thousand = data2.groupby('entity')['hospital_beds_per_thousand'].first().reset_index()

top_10_hosp_beds = hosp_beds_per_thousand.nlargest(10, 'hospital_beds_per_thousand')
bottom_10_hosp_beds = hosp_beds_per_thousand.nsmallest(10, 'hospital_beds_per_thousand')

# 设置图形大小
plt.figure(figsize=(15, 10))
sns.barplot(data=top_10_hosp_beds, x='entity', y='hospital_beds_per_thousand', color='blue')
plt.title('Top 10 Countries by Hospital Beds per Thousand')
plt.xlabel('entity')
plt.ylabel('Hospital Beds per Thousand')
plt.xticks(rotation=45)
plt.savefig(savepath + 'hosp_beds_per_thousand_top_10.png')
plt.show()

# 设置图形大小
plt.figure(figsize=(15, 10))
sns.barplot(data=bottom_10_hosp_beds, x='entity', y='hospital_beds_per_thousand', color='red')
plt.title('Bottom 10 Countries by Hospital Beds per Thousand')
plt.xlabel('entity')
plt.ylabel('Hospital Beds per Thousand')
plt.xticks(rotation=45)
plt.savefig(savepath + 'hosp_beds_per_thousand_bottom_10.png')
plt.show()


'''
分析每个国家的医院 ICU 病人数量
'''
data3 = data.copy()
icu_patients = data3.groupby(['entity', 'year', 'month', 'day'])['icu_patients'].first().reset_index()
country_total_icu = icu_patients.groupby('entity')['icu_patients'].sum()
icu_patients['date'] = pd.to_datetime(icu_patients[['year', 'month', 'day']])
top_5_icu = list(country_total_icu.nlargest(5).index)
bottom_5_icu = list(country_total_icu.nsmallest(5).index)

top_5_icu_data = icu_patients[icu_patients['entity'].isin(top_5_icu)]
bottom_5_icu_data = icu_patients[icu_patients['entity'].isin(bottom_5_icu)]

# 设置图形大小
plt.figure(figsize=(10, 6))
sns.lineplot(x='date', y='icu_patients', hue='entity', data=top_5_icu_data)
plt.title('ICU Patients by Country top 5')
plt.xlabel('Day')
plt.ylabel('Number of ICU Patients')
plt.xticks(rotation=45)
plt.savefig(savepath + 'icu_patients_data_top_5.png')
plt.show()

# 设置图形大小
plt.figure(figsize=(10, 6))
sns.lineplot(x='date', y='icu_patients', hue='entity', data=bottom_5_icu_data)
plt.title('ICU Patients by Country bottom 5')
plt.xlabel('Day')
plt.ylabel('Number of ICU Patients')
plt.xticks(rotation=45)
plt.savefig(savepath + 'icu_patients_data_bottom_5.png')
plt.show()

'''
分析每个国家的每一百万人对医疗资源的占用情况
'''
data4 = data.copy()
per_million_indicators = ['Daily ICU occupancy per million', 'Daily hospital occupancy per million']
icu_occupancy_per_million = data[data['indicator'].isin(['Daily ICU occupancy per million'])]
hospital_occupancy_per_million = data[data['indicator'].isin(['Daily hospital occupancy per million'])]

# 计算每个国家的指标统计信息
country_icu_stats = icu_occupancy_per_million.groupby('entity')['value'].agg(['min', 'max', 'mean', 'median'])
country_hospital_stats = hospital_occupancy_per_million.groupby('entity')['value'].agg(['min', 'max', 'mean', 'median'])

# 显示子数据表格
print(country_icu_stats)
print(country_hospital_stats)

# 找到均值和中位数最大的十个实体名称
top_mean_icu_entities = country_icu_stats.nlargest(10, 'mean').index
top_median_icu_entities = country_icu_stats.nlargest(10, 'median').index

top_mean_hosp_entities = country_hospital_stats.nlargest(10, 'mean').index
top_median_hosp_entities = country_hospital_stats.nlargest(10, 'median').index

# 取最大均值和最大中位数的实体的交集
common_icu_entities = list(set(top_mean_icu_entities) & set(top_median_icu_entities))
common_hosp_entities = list(set(top_mean_hosp_entities) & set(top_median_hosp_entities))
print(common_icu_entities)
print(common_hosp_entities)

# 获取每个实体的每天的数据
icu_data = icu_occupancy_per_million[icu_occupancy_per_million['entity'].isin(common_icu_entities)]
hospital_data = hospital_occupancy_per_million[hospital_occupancy_per_million['entity'].isin(common_hosp_entities)]

# 设置图形大小
plt.figure(figsize=(10, 6))
sns.lineplot(x='date', y='value', hue='entity', data=icu_data)
plt.title('ICU Occupancy per Million by Entity')
plt.xlabel('Day')
plt.ylabel('ICU Occupancy per Million')
plt.xticks(rotation=45)
plt.savefig(savepath + 'percent_patients_data_icu_stats.png')
plt.show()

# 设置图形大小
plt.figure(figsize=(10, 6))
sns.lineplot(x='date', y='value', hue='entity', data=hospital_data)
plt.title('Hospital Occupancy per Million by Entity')
plt.xlabel('Day')
plt.ylabel('Hospital Occupancy per Million')
plt.xticks(rotation=45)
plt.savefig(savepath + 'percent_patients_data_hospital_stats.png')
plt.show()

