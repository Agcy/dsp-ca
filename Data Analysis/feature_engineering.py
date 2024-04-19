## for data
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
## for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data_management import data_manage

# 加载数据集
datapath = '../dataset/clean_data/clean_merged_data.csv'
savepath = '../outputImg/hospital_data/'
data = data_manage(datapath)
data1 = data.copy()
print(data1.head())

# 按日期降序排序
data_sorted = data1.sort_values('date', ascending=False)

# 获取每个国家的最新数据
latest_data = data_sorted.groupby('entity').first().reset_index()

latest_data = latest_data.rename(columns={"total_deaths": "Y"})

subset1 = latest_data.copy()

#Univariate plot
bc1 = sns.countplot(x='Y', data=subset1)
plt.xlabel('New Deaths Number')
plt.title('Daily Deaths rates ')
plt.show()

#count of Y variable
print('count of Y Variable')
c1 = subset1.groupby('Y').size()
print(c1)

#percentages for Y variable
print('percentages of Y variable counts')
p1 = subset1.groupby('Y').size() * 100 / len(subset1)
print(p1)

#Distplot for variable AGE
#numerical

bc2 = sns.displot(subset1.hospital_beds_per_thousand.dropna(), kde=True)
bc2.set_titles('Hospital beds per thousand distribution')
bc2.set_axis_labels('hospital beds distribute', 'Count')
plt.show()

bc2 = sns.displot(subset1.population.dropna(), kde=True)
bc2.set_titles('Population distribution')
bc2.set_axis_labels('population distribution', 'Count')
plt.show()

#Create a figure of a distplot and a boxplot for age
#Create a figure of a distplot and a boxplot for age
fig, ax = plt.subplots(1, 2)
fig.suptitle('Population distribution and outliers')
ax[0].title.set_text('distribution')
sns.histplot(subset1['population'].dropna(), kde=True, ax=ax[0])
plt.xlabel('population')
plt.title('')
plt.xticks(rotation=45)

ax[1].title.set_text('Outliers')
tmp_dtf = pd.DataFrame(subset1['population'])
#tmp_dtf['AGE'] = np.log(tmp_dtf['AGE'])
tmp_dtf.boxplot(column='population', ax=ax[1])
plt.show()

# subset1_optimized = subset1.copy()

#Bivariate plots
fig, ax = plt.subplots(nrows=1, ncols=3,  sharex=False, sharey=False, figsize=(8, 6))
fig.suptitle('hospital_beds_per_thousand   vs   total Deaths', fontsize=20)

### distribution
ax[0].title.set_text('density')
sns.histplot(subset1, x="hospital_beds_per_thousand", multiple="stack", hue='Y', element="step", kde=True, ax=ax[0])
ax[0].grid(True)

ax[1].title.set_text('outliers')
sns.boxplot(x=subset1['Y'], y=subset1['hospital_beds_per_thousand'], data=subset1, whis=np.inf, ax=ax[1])
ax[1].set_xlabel('hospital_beds_per_thousand')
ax[1].grid(True)

plt.show()
