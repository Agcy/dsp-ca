import pandas
import numpy
import geopandas as gpd
import matplotlib.pyplot as plt

#LOAD DATA
hosp_data = pandas.read_csv(
  '../dataset/covid-hospitalizations.csv',
  low_memory=False
)

death_data = pandas.read_csv(
  '../dataset/owid-covid-data.csv',
  low_memory=False
)

# Conduct descriptive statistics
desc_stats_hosp = hosp_data.describe()
desc_stats_death = death_data.describe()

hosp_data['date'] = pandas.to_datetime(hosp_data['date'])

hosp_data['year'] = hosp_data['date'].dt.year
hosp_data['month'] = hosp_data['date'].dt.month
hosp_data['day'] = hosp_data['date'].dt.day

missing_indicator = hosp_data['indicator'].isnull().sum()
missing_value = hosp_data['value'].isnull().sum()

hosp_data['simple_indicator'] = hosp_data['indicator'].replace({
    'Daily ICU occupancy': 'ICU',
    'Daily ICU occupancy per million': 'ICU_per_million'
})

high_occupancy_threshold = 100
hosp_data['high_icu_occupancy'] = hosp_data.apply(
    lambda x: 'High' if x['indicator'] == 'Daily ICU occupancy' and x['value'] > high_occupancy_threshold else 'Normal',
    axis=1
)

hosp_data['value_quartile'] = pandas.qcut(hosp_data['value'], 4, labels=False)

hosp_data.head(), missing_indicator, missing_value

print('the count of hospitalization in country Australia', hosp_data[hosp_data['entity'] == 'Australia'].value_counts())
print(hosp_data["entity"].value_counts(sort=True))

# 对感兴趣的变量运行频率 - 以 death_data 的 'location' 为例
print('the count of death in Taiwan', death_data[death_data['location'] == 'Taiwan'].value_counts())

# 重新编码分类变量中的值 - 以 death_data 的 'continent' 为例
death_data['continent'] = death_data['continent'].replace({'North America': 'NA', 'Europe': 'EU'})

# 使用现有数据创建辅助变量 - 基于 death_data 中的 'total_cases' 和 'total_deaths' 创建 'mortality_rate'
death_data['mortality_rate'] = death_data['total_deaths'] / death_data['total_cases']

# 使用 qcut 进行分组或分类的变量 - 对 death_data 的 'total_deaths' 进行分组
death_data['death_quantile'] = pandas.qcut(death_data['total_deaths'], 4, labels=['Low', 'Medium', 'High', 'Very High'])

# 操作日期变量
death_data['date'] = pandas.to_datetime(death_data['date'])
death_data["year"] = death_data['date'].apply(lambda x: x.year)
death_data["month"] = death_data['date'].apply(lambda x: x.month)
death_data["day"] = death_data['date'].apply(lambda x: x.day)

# 打印一些转换后的数据以验证
print(death_data[['date', 'year', 'month', 'day', 'continent', 'mortality_rate', 'death_quantile']].head())

# 读取地图数据
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# 将您的数据集与世界地图数据合并，假设您的数据集中有一个国家代码字段与世界地图数据中的iso_a3字段相匹配
# 此处以 hosp_data 为例，假设 hosp_data 中的 iso_code 字段为国家代码
merged = world.set_index('iso_a3').join(hosp_data.set_index('iso_code'))

# 绘制地图
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
merged.plot(column='value', ax=ax, legend=True,
            legend_kwds={'label': "ICU 占用率", 'orientation': "horizontal"})
plt.show()

# Begin creating univariate graphs for categorical variables
# Example: Bar chart for 'entity' column in hosp_data
hosp_data['entity'].value_counts().plot(kind='bar')
plt.title('Frequency of Different Entities in Hospital Data')
plt.xlabel('Entity')
plt.ylabel('Frequency')
plt.show()

# Begin creating univariate graphs for numerical variables
# Example: Histogram for 'value' column in hosp_data
hosp_data['value'].hist(bins=50)
plt.title('Distribution of ICU Occupancy Values')
plt.xlabel('ICU Occupancy Value')
plt.ylabel('Frequency')
plt.show()

documentation = """
Descriptive Statistics:
----------------------
Hospitalization Data:
{desc_stats_hosp}

Death Data:
{desc_stats_death}

Graphs:
-------
The bar chart and histogram have been created and displayed to understand the frequency distribution of the entities and the distribution of ICU occupancy values, respectively.

Variables:
----------
Assuming 'entity' and 'date' are explanatory variables, and 'value' in hospitalization data is a response variable, as we might want to predict the ICU occupancy based on these factors.

Summary Report:
---------------
A detailed summary report is underway with the findings and insights from the descriptive statistics and graphs.
""".format(desc_stats_hosp=desc_stats_hosp, desc_stats_death=desc_stats_death)

print(documentation)