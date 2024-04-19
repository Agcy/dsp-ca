import pandas as pd


def load_data(hosp_data_path, death_data_path):
    """加载数据并进行初步清洁"""
    hosp_cols = ['entity', 'date', 'iso_code', 'indicator', 'value']
    death_cols = ['date', 'iso_code', 'continent', 'total_deaths', 'new_deaths', 'hosp_patients',
                  'icu_patients', 'hospital_beds_per_thousand', 'population', 'population_density']
    hosp_data = pd.read_csv(hosp_data_path, usecols=hosp_cols)
    death_data = pd.read_csv(death_data_path, usecols=death_cols)
    merged_data = pd.merge(hosp_data, death_data, on=['date', 'iso_code'], how='inner')
    # 检查缺失值
    # 处理缺失值 - 这里我们选择删除含缺失值的行
    merged_data.dropna(inplace=True)

    # # 数据预处理 - 日期处理
    # merged_data['date'] = pd.to_datetime(hosp_data['date'])
    #
    # # 提取年、月、日
    #
    # merged_data['year'] = death_data['date'].dt.year
    # merged_data['month'] = death_data['date'].dt.month
    # merged_data['day'] = death_data['date'].dt.day

    return merged_data


def save_clean_data(merged_data, clean_merged_data_path):
    """保存清洁后的数据"""
    merged_data.to_csv(clean_merged_data_path, index=False)


if __name__ == "__main__":
    raw_hosp_data_path = '../dataset/covid-hospitalizations.csv'
    raw_death_data_path = '../dataset/owid-covid-data.csv'
    clean_merged_data_path = '../dataset/clean_data/clean_merged_data.csv'
    merged_data = load_data(raw_hosp_data_path, raw_death_data_path)
    save_clean_data(merged_data, clean_merged_data_path)
