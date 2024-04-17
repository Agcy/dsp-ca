import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd


def load_data(file_path):
    """加载数据"""
    return pd.read_csv(file_path)


def describe_data(data):
    """打印数据的描述性统计量"""
    print(data.describe())


def plot_histogram(data, variable):
    """绘制变量的直方图"""
    plt.figure(figsize=(10, 6))
    sns.histplot(data[variable], kde=True)
    plt.title('Distribution of ' + variable)
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.show()


def plot_time_series_chart(data, variable):
    """绘制变量的时间序列图"""
    data['date'] = pd.to_datetime(data['date'])  # 确保日期列为datetime类型
    data.set_index('date', inplace=True)
    data[variable].plot()
    plt.title('Total Deaths Over Time')
    plt.xlabel('Date')
    plt.ylabel(variable)
    plt.show()


def plot_scatter(data, var1, var2):
    """绘制两个变量的散点图"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=var1, y=var2, data=data)
    plt.title('Scatter Plot of {} vs {}'.format(var1, var2))
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.show()


def plot_boxplot(data, variable):
    """绘制变量的箱型图"""
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=variable, data=data)
    plt.title('Box Plot of ' + variable)
    plt.ylabel(variable)
    plt.show()


def plot_bar(data, variable1, variable2):
    plt.figure(figsize=(15, 12))
    continent_total_deaths = data.groupby(variable1)[variable2].sum().reset_index()
    sns.barplot(data=continent_total_deaths, x=variable1, y=variable2)
    plt.title('Total Deaths by Continent')
    plt.xlabel(variable1)
    plt.ylabel(variable2)
    plt.xticks(rotation=45)
    plt.show()


def plot_heatmap(data, variable1, variable2, variable3, variable4):
    correlation_matrix = data[[variable1, variable2, variable3, variable4]].corr()
    sns.heatmap(correlation_matrix, annot=True)
    plt.title('Correlation Matrix')
    plt.show()


if __name__ == "__main__":
    clean_data_path = '../dataset/clean_data/clean_merged_data.csv'
    data = load_data(clean_data_path)
    describe_data(data)

    plot_histogram(data, 'new_deaths')
    # plot_time_series_chart(data, 'total_deaths')
    plot_scatter(data, 'continent', 'total_deaths')
    plot_boxplot(data, 'total_deaths')
    plot_bar(data, 'continent', 'total_deaths')
    plot_heatmap(data, 'total_deaths', 'new_deaths', 'icu_patients', 'hosp_patients')

    # 过滤出澳大利亚的数据并且指标为'Daily hospital occupancy per million'
    australia_data = data[
        (data['entity'] == 'Australia') & (data['indicator'] == 'Daily hospital occupancy per million')]

    # print("source data "+data.columns)
    # print("aus data "+australia_data.columns)
    # 将'date'列转换为日期格式，并对数据按日期排序
    australia_data['date'] = pd.to_datetime(australia_data['date'])
    australia_data.sort_values('date', inplace=True)

    plt.figure(figsize=(12, 6))
    plt.plot(australia_data['date'], australia_data['value'], marker='o')
    plt.title('Daily Hospital Occupancy per Million in Australia')
    plt.xlabel('Date')
    plt.ylabel('Daily Hospital Occupancy per Million')
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
    plt.show()

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # 加载你的数据（确保'entity'列与世界地图上的'iso_a3'或'name'列相匹配）
    # data = pd.read_csv('path_to_your_file.csv')

    # 取得最新日期的数据
    latest_data = data[data['date'] == data['date'].max()]

    # 将数据合并到世界地图上
    merged = world.set_index('name').join(latest_data.set_index('entity'))

    # 画出带有total_deaths数据的地图
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    merged.plot(column='total_deaths', ax=ax, legend=True,
                legend_kwds={'label': "Total Deaths by Country",
                             'orientation': "horizontal"})
    plt.show()
