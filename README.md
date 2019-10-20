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

## 2 - Launch the demo
Use the following command to launch Streamlit dashboard on __localhost:8501__: 
```bash
streamlit run src/dashboard.py
```

Note: The dataset used is the train set from  __Bike Sharing Demand__'s **train.csv**. It was downloaded from [Kaggle](https://www.kaggle.com/c/bike-sharing-demand).  
