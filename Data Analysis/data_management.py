import pandas as pd


def data_manage(datapath):
    covid_data = pd.read_csv(datapath)
    # divide the date object into three columns of year, month, and day data
    covid_data['date'] = pd.to_datetime(covid_data['date'])
    covid_data["year"] = covid_data['date'].apply(lambda x: x.year)
    covid_data["month"] = covid_data['date'].apply(lambda x: x.month)
    covid_data["day"] = covid_data['date'].apply(lambda x: x.day)

    # counting classified variable frequency
    # print(covid_data.columns)
    print(covid_data['continent'].value_counts())
    print(covid_data['indicator'].value_counts())
    print(covid_data.isnull().sum())

    # number the classified variable
    # covid_data['continent'] = covid_data['continent'].replace({'North America': 'NA', 'Europe': 'EU'})

    # create secondary variable
    covid_data['mortality_rate'] = covid_data['total_deaths'] / covid_data['population']

    # grouping variable
    covid_data['deaths_quartile'] = pd.qcut(covid_data['total_deaths'], 4, labels=False)

    print(covid_data.columns)

    return covid_data


if __name__ == '__main__':
    datapath = '../dataset/clean_data/clean_merged_data.csv'
    data_manage(datapath)
