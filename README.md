# demo_streamlit
This is a demo for Streamlit, for my article on OCTO blog https://blog.octo.com/creer-une-web-app-interactive-en-10min-avec-streamlit/.

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

![Web-app demo](images/demo.gif)

## Notes
The dataset used is the train set from  __Bike Sharing Demand__'s **train.csv**. It was downloaded from [Kaggle](https://www.kaggle.com/c/bike-sharing-demand).  

The code can also be directly run from [Gist](https://gist.github.com/AurelienMassiot/b3070dab9e31dd119242648b4d27c9b4) remotely. For this you need to install Streamlit:
```bash
pip install streamlit
```
Then, launch Streamlit on the code shared on Gist:
```bash
streamlit run https://gist.githubusercontent.com/AurelienMassiot/b3070dab9e31dd119242648b4d27c9b4/raw/e35965ebe409d31fcfe59d5574e33641a2b43728/dashboard.py
``` 
