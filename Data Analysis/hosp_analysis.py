from data_management import data_manage
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据集
datapath = '../dataset/clean_data/clean_merged_data.csv'
savepath = '../dataset/image_data/'
data = data_manage(datapath)
data1 = data.copy()