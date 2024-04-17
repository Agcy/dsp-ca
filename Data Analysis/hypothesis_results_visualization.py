from data_management import data_manage
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind, mannwhitneyu
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

datapath = '../dataset/clean_data/clean_merged_data.csv'
savepath = '../outputImg/hospital_data/'
data = data_manage(datapath)
data1 = data.copy()

data_sorted = data1.sort_values('date', ascending=False)

latest_data = data_sorted.groupby('entity').first().reset_index()

x = latest_data['mortality_rate']
y = latest_data['hospital_beds_per_thousand']

corr, p_value = pearsonr(x, y)

print('Pearson correlation: ', corr)
print('p-value: ', p_value)

t_statistic, p_value_t = ttest_ind(latest_data['total_deaths'], latest_data['mortality_rate'])

u_statistic, p_value_u = mannwhitneyu(latest_data['total_deaths'], latest_data['mortality_rate'])

print('\n T-test Result')
print('T statistic: ', t_statistic)
print('p-value: ', p_value_t)

print('\n Mann-Whitney U-test Result')
print('U statistic: ', u_statistic)
print('p-value: ', p_value_u)
