import calendar
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly_express as px
import streamlit as st


# %%

@st.cache
def load_data():
    bikes_data_path = Path() / 'data/bike_sharing_demand_train.csv'
    data = pd.read_csv(bikes_data_path)
    return data


def preprocess_data(dataframe):
    dataframe['date'] = dataframe['datetime'].apply(lambda x: x.split()[0])
    dataframe['hour'] = dataframe['datetime'].apply(lambda x: x.split()[1].split(':')[0])
    dataframe['weekday'] = dataframe['date'].apply(
        lambda date_string: calendar.day_name[datetime.strptime(date_string, '%Y-%m-%d').weekday()])
    dataframe['month'] = dataframe['date'].apply(
        lambda date_string: calendar.month_name[datetime.strptime(date_string, '%Y-%m-%d').month])
    dataframe['season'] = dataframe['season'].map({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'})
    return dataframe


st.title('Bike Sharing demand')
df = load_data()
df_preprocessed = preprocess_data(df.copy())
st.write(df_preprocessed)

# %% barplot
st.subheader('Barplot')
mean_counts_by_hour = pd.DataFrame(df_preprocessed.groupby(['hour', 'season'], sort=True)['count'].mean()).reset_index()
fig1 = px.bar(mean_counts_by_hour, x='hour', y='count', color='season', height=400)
barplot_chart = st.write(fig1)

# %% timeseries
st.subheader('Timeseries')
df_preprocessed['datetime'] = pd.to_datetime(df_preprocessed['datetime'])
fig2 = px.line(df_preprocessed, x='datetime', y='temp')
ts_chart = st.plotly_chart(fig2)

# %% boxplots
st.subheader('Boxplots')
fig3 = px.box(df_preprocessed, x="weekday", y="count", color="season", notched=True)
boxplot_chart = st.plotly_chart(fig3)