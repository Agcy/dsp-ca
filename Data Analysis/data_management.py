import pandas as pd

def data_management(datapath):
    covid_data = pd.read_csv(datapath)
    # divide the date object into three columns of year, month, and day data
    covid_data['date'] = pd.to_datetime(covid_data['date'])
    covid_data["year"] = covid_data['date'].apply(lambda x: x.year)
    covid_data["month"] = covid_data['date'].apply(lambda x: x.month)
    covid_data["day"] = covid_data['date'].apply(lambda x: x.day)
    print(covid_data.columns)
    return covid_data

if __name__ == '__main__':
    datapath = '../dataset/clean_data/clean_merged_data.csv'
    data_management(datapath)
