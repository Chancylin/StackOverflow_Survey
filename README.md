## Table of Contents
1. [Project Description](#project_desc)
2. [CRISP-DM For Job Satisfaction Prediction](#CRISP-DM)
3. [File Structure](#fileStructure)
4. [Requirements](#requirements)
5. [Acknowledgements](#ack)

### 1. Project Description <a name="project_desc"></a>
In this repository, we use [Stack Overflow Annual Developer Survey data](https://insights.stackoverflow.com/survey) 
to do some exploratory analyses of data jobs (Data Scientist, Data Analyst, Data Engineer). 
Analyses include facts on salary, job satisfaction, change of the data jobs (2019 v.s. 2020). 
Machine learning modelling has also been used to predict job satisfaction.

A medium post about the analyses results can be found [here](https://lcxustc.medium.com/salary-satisfaction-trend-of-data-jobs-f47bdf72afa3).

### 2. CRISP-DM For Job Satisfaction Prediction <a name="CRISP-DM"></a>
In particular, for the job satisfaction prediction, the procedure of [CRISP-DM](https://www.datascience-pm.com/crisp-dm-2/) for data mining has been applied.
#### - Business Problem
We are interested in understanding job satisfaction for data jobs, and try to identify some important factors. This
 may be useful, as example, for a Career Advice & Coach Company to better plan career development for their
  customers, such as job seekers to switch to the data field or data professionals to make the next step in their
   career path. 

#### - Data Understanding
The Stack Overflow Survey Data is a questionnaire designed for people with general developer background. It
 covers questions about personal background such as age, gender, education level, and job-related questions such as
  salary, work hours, job types, job satisfaction, etc. We focus on responders who have a data-related jobs and use the
   responses to job satisfaction as the entry point to understand what contribute to job satisfaction. Exploratory
    data analysis has been applied to get some preliminary insight into job satisfaction. Then modelling technique is applied.
 
#### - Data Preparation
To prepare the raw survey data for modelling purpose, we need to first extract responses that are associated with data-related jobs. Standard data preparation procedure has been applied, mainly including:

* select the subset of data of interest and transform raw data properly;
* feature selection;
* missing data imputation;
* categorical data encoding.
 

 There are 10,372 data instances and 38 features after data preparation (54 feature columns after one-hot encoding).
 
#### - Modelling
There are five possible responses to job satisfaction, so it is a multi-classification problem. Data has been split
into training and test set. An untuned Gaussian Naive Bayes has been tried as a baseline model. And then grid search with cross-validation technique is applied to tune XGBoost model. Average ROC-AUC score has been used as the main metric for hyperparameter tuning and model performance evaluation, while other metrics such as Log-Loss, Accuracy, average Precision (macro), average Recall (macro) and Confusion Matrix have been used together to give a comprehensive evaluation of the model performance. 

#### - Evaluation
A set of metrics have been used to evaluate the model performance. Although there exist the overfitting issue to
some extent, the model performance is considered acceptable and the model has a reasonable predictive power. We also applied some explainability technique (e..g, [SHAP](https://github.com/slundberg/shap)) to get more insight of the modelling result, such as key drivers (e.g., Salary, OnboardExperience, Age, YearsCode, CompanySize) and dependence relationship w.r.t. predicting job satisfaction.
   
#### - Deployment
This project doesn't cover the deployment part, but with the trained model, one can productionize it, such as
 integrating it into an App, and score new data instance. 

### 3. File Structure <a name="fileStructure"></a>
    
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

### 4. Requirements <a name="requirements"></a>
Python 3.7.5 is used in this project. To install the required packages, you can run
```Bash
conda install --file requirements.txt
```
in your conda environment
or use pip install
```Bash
pip install -r requirements.txt
```

### 5. Acknowledgements <a name="ack"></a>
Credit goes to Stack Overflow for the data. Note that this project is a showcase of data analysis and machine learning practice. Any findings and results are subject to futher investigation. Feel free to use the code here as you would like.
