import datetime
from pathlib import Path

import pandas as pd
import plotly_express as px
import shap
import numpy as np
import streamlit as st
from feature_engineering import preprocess_data
from sklearn.ensemble import RandomForestRegressor
from time import sleep


# %%
@st.cache
def load_data():
    bikes_data_path = Path() / 'data/bike_sharing_demand_train.csv'
    data = pd.read_csv(bikes_data_path)
    return data


st.title('Bike Sharing demand')
df = load_data()
df_preprocessed = preprocess_data(df.copy())
st.write(df_preprocessed)

st.title('Data exploration')
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
categories_count = ['casual', 'registered', 'count']
chosen_count = st.sidebar.selectbox(
    'Which counts for boxplots?',
    categories_count
)

fig3 = px.box(df_preprocessed, x='weekday', y=chosen_count, color='season', notched=True)
boxplot_chart = st.plotly_chart(fig3)

st.title('Modelization')

# %% Modelization
X = df_preprocessed[['temp', 'humidity', 'windspeed']]
y = df_preprocessed['count']
model_rf = RandomForestRegressor(max_depth=2, n_estimators=10)
model_rf.fit(X, y)

# %% Shap
#explainer = shap.TreeExplainer(model_rf)
#shap_values = explainer.shap_values(X)

# shap_summary_chart = st.write(shap.summary_plot(shap_values, X))
#st.write(shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True))
# ne marche pas, même avec l'option matplotlib=True

# %% Stream new data


@st.cache
def generate_data(df, column, n_rows):
    freq = '1h'
    time_begin_new_data = df['datetime'].max() + datetime.timedelta(hours=1)
    time_end_new_data = df['datetime'].max() + datetime.timedelta(hours=n_rows)
    times = pd.date_range(time_begin_new_data, time_end_new_data, freq=freq)
    random_numbers = np.random.uniform(df[column].min(), df[column].max(),
                                       size=(len(times),))
    new_df = pd.DataFrame({'datetime': times,
                           column: random_numbers,
                           'predicted': True})
    return new_df


def add_line(df, new_row_df):
    return pd.concat([df, new_row_df], axis=0).reset_index(drop=True)


def animate(df, column, the_plot):
    fig = px.line(df, x='datetime', y=column, color='predicted')
    the_plot.plotly_chart(fig)


# %% new data
chosen_sensor = 'temp'
n_rows = 50
new_df = generate_data(df_preprocessed, chosen_sensor, 15)
df_for_predictions = df_preprocessed.copy()
df_for_predictions['predicted'] = False
fig = px.line(df_for_predictions.tail(n_rows), x='datetime', y=chosen_sensor, color='predicted')
the_plot = st.plotly_chart(fig)
new_row_info = st.empty()

if st.sidebar.checkbox('Stream and predict on new data'):
    bar = st.progress(0)
    for i in range(11):
        # get new line
        new_row = pd.DataFrame(new_df.iloc[[i]])
        new_row_info.info(f'Received new value: {new_row}')
        # concat data
        df_for_predictions = add_line(df_for_predictions, new_row)
        # animate
        animate(df_for_predictions.tail(n_rows), chosen_sensor, the_plot)
        bar.progress(i * 10)
        # wait
        sleep(0.1)

