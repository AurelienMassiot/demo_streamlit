import datetime
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import plotly_express as px
import streamlit as st
from feature_engineering import preprocess_data
from sklearn.ensemble import RandomForestRegressor


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
# %% barplots
st.subheader('Barplots')
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
X = df_preprocessed[['temp', 'humidity']]
y = df_preprocessed['count']
model_rf = RandomForestRegressor(max_depth=2, n_estimators=10)
model_rf.fit(X, y)


# %% Shap
# explainer = shap.TreeExplainer(model_rf)
# shap_values = explainer.shap_values(X)

# shap_summary_chart = st.write(shap.summary_plot(shap_values, X))
# st.write(shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True))
# ne marche pas, même avec l'option matplotlib=True

# %% Stream new data


@st.cache
def generate_new_row(df):
    time_end_new_data = df['datetime'].max() + datetime.timedelta(hours=1)
    random_number_temp = np.random.uniform(df['temp'].min(), df['temp'].max(),
                                           size=(1), )
    random_number_humidity = np.random.uniform(df['humidity'].min(), df['humidity'].max(),
                                               size=(1), )
    new_df = pd.DataFrame({'datetime': [time_end_new_data],
                           'temp': random_number_temp,
                           'humidity': random_number_humidity,
                           'predicted': [True]})
    return new_df


def add_row(df, new_row_df):
    return pd.concat([df, new_row_df], axis=0).reset_index(drop=True)


def generate_new_prediction(df, row, model):
    time_end_new_data = df['datetime'].max() + datetime.timedelta(hours=1)
    X_pred = row[['temp', 'humidity']]
    y_pred = model.predict(X_pred)
    new_df = pd.DataFrame({'datetime': [time_end_new_data],
                           'count': y_pred,
                           'predicted': [True]})
    return new_df


def animate(df, column, the_plot):
    fig = px.line(df, x='datetime', y=column, color='predicted')
    the_plot.plotly_chart(fig)


# %% new data
column = 'count'
n_rows_to_display = 50
df_for_predictions = df_preprocessed.copy()
df_for_predictions['predicted'] = False
fig = px.line(df_for_predictions.tail(n_rows_to_display), x='datetime', y=column, color='predicted')
the_plot = st.plotly_chart(fig)
new_row_info = st.empty()
predicted_row_warning = st.empty()

if st.sidebar.checkbox('Stream and predict on new data'):
    bar = st.progress(0)
    for i in range(11):
        # get new row
        new_row = generate_new_row(df_for_predictions)
        new_row_info.info(f'Received new values: \n'
                          f'temperature={np.round(new_row["temp"].values[0], 2)} - \n'
                          f'humidity={np.round(new_row["humidity"].values[0], 2)} \n')
        # predict
        new_prediction = generate_new_prediction(df_for_predictions, new_row, model_rf)
        predicted_row_warning.warning(f'Predicted count: {np.round(new_prediction["count"].values[0], 2)}')
        # concatenate predicted row
        df_for_predictions = add_row(df_for_predictions, new_prediction)
        # animate
        animate(df_for_predictions.tail(n_rows_to_display), column, the_plot)
        bar.progress(i * 10)
        # wait
        sleep(0.1)