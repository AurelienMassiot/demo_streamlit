import calendar
from datetime import datetime


def preprocess_data(dataframe):
    dataframe['date'] = dataframe['datetime'].apply(lambda x: x.split()[0])
    dataframe['hour'] = dataframe['datetime'].apply(lambda x: x.split()[1].split(':')[0])
    dataframe['weekday'] = dataframe['date'].apply(
        lambda date_string: calendar.day_name[datetime.strptime(date_string, '%Y-%m-%d').weekday()])
    dataframe['month'] = dataframe['date'].apply(
        lambda date_string: calendar.month_name[datetime.strptime(date_string, '%Y-%m-%d').month])
    dataframe['season'] = dataframe['season'].map({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'})
    return dataframe