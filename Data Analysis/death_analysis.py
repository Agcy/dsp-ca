from data_management import data_manage
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据集
datapath = '../dataset/clean_data/clean_merged_data.csv'
savepath = '../dataset/image_data/'
data = data_manage(datapath)
data1 = data.copy()
# 按日期降序排序
data_sorted = data1.sort_values('date', ascending=False)

# 获取每个国家的最新数据
latest_data = data_sorted.groupby('entity').first().reset_index()

# 获取前十名和后十名数据
top_10 = latest_data.nlargest(10, 'total_deaths')
bottom_10 = latest_data.nsmallest(10, 'total_deaths')

# 获取前十名和后十名的国家死亡人口比例
top_10_percent = latest_data.nlargest(10, 'mortality_rate')
bottom_10_percent = latest_data.nsmallest(10, 'mortality_rate')
# 设置图形大小
plt.figure(figsize=(15, 10))

# 绘制前十名直方图
# 绘制前十名直方图
sns.barplot(data=top_10, x='entity', y='total_deaths', color='blue')

# 添加数值标签
for i, value in enumerate(top_10['total_deaths']):
    plt.text(i, value, str(int(value)), ha='center', va='bottom')

# 设置图形标题和轴标签
plt.title('Top 10 Countries by Total Deaths (Latest Data)')
plt.xlabel('entity')
plt.ylabel('Total Deaths')

# 旋转 x 轴标签
plt.xticks(rotation=45)

# 显示图形
plt.savefig(savepath+'top_10_number.png')
plt.show()


# 死亡人数最少的十个国家
# 设置图形大小
plt.figure(figsize=(15, 10))

# 绘制后十名直方图
sns.barplot(x='entity', y='total_deaths', data=bottom_10, color='red')

# 添加数值标签
for i, value in enumerate(bottom_10['total_deaths']):
    plt.text(i, value, str(int(value)), ha='center', va='bottom')

# 设置图形标题和轴标签
plt.title('Bottom 10 Countries by Total Deaths (Latest Data)')
plt.xlabel('entity')
plt.ylabel('Total Deaths')

# 旋转 x 轴标签
plt.xticks(rotation=45)

plt.savefig(savepath+'bottom_10_number.png')
# 显示图形
plt.show()


# 比例部分
# 前十
plt.figure(figsize=(15, 10))
sns.barplot(data=top_10_percent, x='entity', y='mortality_rate', color='blue')

for i, value in enumerate(top_10_percent['mortality_rate']):
    plt.text(i, value, str(float(value)), ha='center', va='bottom')

plt.title('Top 10 Countries by Mortality Rate * 1000 (Latest Data)')
plt.xlabel('entity')
plt.ylabel('Mortality Rate')
plt.xticks(rotation=45)
plt.savefig(savepath+'top_10_percent.png')
plt.show()

plt.figure(figsize=(15, 10))
sns.barplot(data=bottom_10_percent, x='entity', y='mortality_rate', color='red')
for i, value in enumerate(bottom_10_percent['mortality_rate']):
    plt.text(i, value, str(float(value)), ha='center', va='bottom')

plt.title('Bottom 10 Countries by Mortality Rate * 1000 (Latest Data)')
plt.xlabel('entity')
plt.ylabel('Mortality Rate')
plt.xticks(rotation=45)
plt.savefig(savepath+'bottom_10_percent.png')
plt.show()