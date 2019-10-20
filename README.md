# demo_streamlit
This is a demo for Streamlit, for the article I'll publish soon :-).

## 1 - Environment creation
Environment is managed by Pipenv. By running the following command, dependencies from Pipfile will be installed:
```bash
pipenv install
```

Then, activate the Pipenv Shell:
```bash
pipenv shell
```

## 2 - Download data
The dataset used is __Bike Sharing Demand__. It can be downloaded on [Kaggle](https://www.kaggle.com/c/bike-sharing-demand).  
Then, put the dataset in `data/` and rename it `bike_sharing_demand_train.csv.


## 3 - Launch the demo
Use the following command to launch Streamlit dashboard on __localhost:8501__: 
```bash
streamlit run src/dashboard.py
```