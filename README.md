## Table of Contents
1. Project Description
2. File Structure
3. Requirements
4. Acknowledgements

### 1. Project Description
In this repository, we use [Stack Overflow Annual Developer Survey data](https://insights.stackoverflow.com/survey) 
to do some exploratory analyses of data jobs (Data Scientist, Data Analyst, Data Engineer). 
Analyses include facts on salary, job satisfaction, change of the data jobs (2019 v.s. 2020). 
Machine learning modelling has also been used to predict job satisfaction.

A medium post about the analyses results can be found [here](https://lcxustc.medium.com/salary-satisfaction-trend-of-data-jobs-f47bdf72afa3).

### 2. File Structure
    
    ├── README.md
    ├── code  # helpers
    │   ├── data_process.py     # helper functions for data processing
    │   └── modelling.py        # helper function for modelling
    ├── data
    │   ├── developer_survey_2019.zip
    │   └── developer_survey_2020.zip
    ├── model
    │   └── xgb_cv_pipeline     # trained xgb GridSearchCV pipeline
    └── notebooks
        └── stackoverflow.ipynb # jupyter notebook, main work of analyses and modelling

### 3. Requirements
Python 3.7.5 is used in this project. To install the required packages, you can run
```Bash
conda install --file requirements.txt
```
in your conda environment
or use pip install
```Bash
pip install -r requirements.txt
```

### 4. Acknowledgements
Credit goes to Stack Overflow for the data. Note that this project is a showcase of data analysis and machine learning practice. Any findings and results are subject to futher investigation. Feel free to use the code here as you would like.
