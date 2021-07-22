# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 01:16:20 2021

@author: Harish
"""

from flask import Flask,request,render_template
import jsonify
import requests
import pickle
import numpy
import sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler


app = Flask(__name__,template_folder='templates')
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    Age = request.form.get('Age')
    BusinessTravel = request.form['BusinessTravel']
    DailyRate = request.form.get('DailyRate')
    Department = request.form['Department']
    DistanceFromHome = request.form.get('DistanceFromHome')
    Education = request.form.get('Education')
    EducationField = request.form['EducationField']
    EnvironmentSatisfaction = request.form.get('EnvironmentSatisfaction')
    Gender = request.form['Gender']
    HourlyRate = request.form.get('HourlyRate')
    JobInvolvement = request.form.get('JobInvolvement')
    MonthlyIncome = request.form.get('MonthlyIncome')
    JobRole = request.form['JobRole']
    JobLevel  = request.form.get('JobLevel')
    JobSatisfaction = request.form.get('JobSatisfaction')
    MaritalStatus = request.form['MaritalStatus']
    NumCompaniesWorked =  request.form.get('NumCompaniesWorked')
    OverTime = request.form['OverTime']
    PercentSalaryHike =  request.form.get('PercentSalaryHike')
    PerformanceRating = request.form.get('PerformanceRating')
    RelationshipSatisfaction = request.form.get('RelationshipSatisfaction')
    StockOptionLevel = request.form.get('StockOptionLevel')
    TotalWorkingYears = request.form.get('TotalWorkingYears')
    TrainingTimesLastYear = request.form.get('TrainingTimesLastYear')
    WorkLifeBalance = request.form.get('WorkLifeBalance')
    YearsAtCompany = request.form.get('YearsAtCompany')
    YearsInCurrentRole = request.form.get('YearsInCurrentRole')
    YearsSinceLastPromotion = request.form.get('YearsSinceLastPromotion')
    YearsWithCurrManager = request.form.get('YearsWithCurrManager')
    
    
    dict = {'Age': int(Age),
            'BusinessTravel': str(BusinessTravel),
            'DailyRate': int(DailyRate),
            'Department': str(Department),
            'DistanceFromHome': int(DistanceFromHome),
            'Education': int(Education),
            'EducationField': str(EducationField),
            'EnvironmentSatisfaction': int(EnvironmentSatisfaction),
            'Gender': str(Gender),
            'HourlyRate': int(HourlyRate),
            'JobInvolvement':int(JobInvolvement),
            'JobLevel': int(JobLevel),
            'JobRole': str(JobRole),
            'JobSatisfaction': int(JobSatisfaction),
            'MaritalStatus': str(MaritalStatus),
            'MonthlyIncome': int(MonthlyIncome),
            'NumCompaniesWorked':int(NumCompaniesWorked),
            'OverTime': str(OverTime),
            'PercentSalaryHike': int(PercentSalaryHike),
            'PerformanceRating': int(PerformanceRating),
            'RelationshipSatisfaction': int(RelationshipSatisfaction),
            'StockOptionLevel': int(StockOptionLevel),
            'TotalWorkingYears': int(TotalWorkingYears),
            'TrainingTimesLastYear': int(TrainingTimesLastYear),
            'WorkLifeBalance': int(WorkLifeBalance),
            'YearsAtCompany': int(YearsAtCompany),
            'YearsInCurrentRole': int(YearsInCurrentRole),
            'YearsSinceLastPromotion': int(YearsSinceLastPromotion),
            'YearsWithCurrManager': int(YearsWithCurrManager)
            }
    df = pd.DataFrame([dict])
        
        
    df.drop(['DailyRate','HourlyRate',
             'EnvironmentSatisfaction','JobSatisfaction',
             'RelationshipSatisfaction'],axis=1,inplace=True)
    df.drop(['JobLevel','TotalWorkingYears','YearsAtCompany',
             'PerformanceRating','YearsInCurrentRole'],
              axis=1,inplace=True)
        
        
    if BusinessTravel == 'Rarely':
        df['BusinessTravel_Travel_Rarely'] = 1
        df['BusinessTravel_Travel_Frequently'] = 0
    elif BusinessTravel == 'Frequently':
        df['BusinessTravel_Travel_Rarely'] = 0
        df['BusinessTravel_Travel_Frequently'] = 1
    else:
        df['BusinessTravel_Travel_Rarely'] = 0
        df['BusinessTravel_Travel_Frequently'] = 0
    df.drop ('BusinessTravel',axis=1,inplace=True)
        
        
    if Department == 'Research & Development':
        df['Department_Research & Development'] = 1
        df['Department_Sales'] = 0
    elif Department == 'Sales':
        df['Department_Research & Development'] = 0
        df['Department_Sales'] = 1
    else:
        df['Department_Research & Development'] = 0
        df['Department_Sales'] = 0
    df.drop('Department',axis=1,inplace = True)
    
    
    if EducationField == 'Life Sciences':
        df['EducationField_Life Sciences'] = 1
        df['EducationField_Marketing'] = 0
        df['EducationField_Medical'] = 0
        df['EducationField_Other'] = 0
        df['EducationField_Technical Degree'] = 0
    elif EducationField == 'Marketing':
        df['EducationField_Life Sciences'] = 0
        df['EducationField_Marketing'] = 1
        df['EducationField_Medical'] = 0
        df['EducationField_Other'] = 0
        df['EducationField_Technical Degree'] = 0
    elif EducationField == 'Medical':
        df['EducationField_Life Sciences'] = 0
        df['EducationField_Marketing'] = 0
        df['EducationField_Medical'] = 1
        df['EducationField_Other'] = 0
        df['EducationField_Technical Degree'] = 0
    elif EducationField == 'Other':
        df['EducationField_Life Sciences'] = 0
        df['EducationField_Marketing'] = 0
        df['EducationField_Medical'] = 0
        df['EducationField_Other'] = 1
        df['EducationField_Technical Degree'] = 0
    elif EducationField == 'Technical Degree':
        df['EducationField_Life Sciences'] = 0
        df['EducationField_Marketing'] = 0
        df['EducationField_Medical'] = 0
        df['EducationField_Other'] = 0
        df['EducationField_Technical Degree'] = 1
    else:
        df['EducationField_Life Sciences'] = 0
        df['EducationField_Marketing'] = 0
        df['EducationField_Medical'] = 0
        df['EducationField_Other'] = 0
        df['EducationField_Technical Degree'] = 0
    df.drop('EducationField',axis=1,inplace = True)
    
    if Gender == 'Male':
        df['Gender_Male'] = 1
    else:
        df['Gender_Male'] = 0
    df.drop('Gender',axis=1,inplace=True)
        
    if OverTime == 'Yes':
        df['OverTime_Yes'] = 1
    else:
        df['OverTime_Yes'] = 0
    df.drop('OverTime',axis=1,inplace=True)
        
        
    if MaritalStatus == 'Married':
        df['MaritalStatus_Married'] = 1
        df['MaritalStatus_Single'] = 0
    elif MaritalStatus == 'Single':
        df['MaritalStatus_Married'] = 0
        df['MaritalStatus_Single'] = 1
    else:
        df['MaritalStatus_Married'] = 0
        df['MaritalStatus_Single'] = 0
    df.drop('MaritalStatus',axis=1,inplace =True)
        
    if JobRole == 'Human Resources':
        df['JobRole_Human Resources'] = 1
        df['JobRole_Laboratory Technician'] = 0
        df['JobRole_Manager'] = 0
        df['JobRole_Manufacturing Director'] = 0
        df['JobRole_Research Director'] = 0
        df['JobRole_Research Scientist'] = 0
        df['JobRole_Sales Executive'] = 0
        df['JobRole_Sales Representative'] = 0
    elif JobRole == 'Laboratory Technician':
        df['JobRole_Human Resources'] = 0
        df['JobRole_Laboratory Technician'] = 1
        df['JobRole_Manager'] = 0
        df['JobRole_Manufacturing Director'] = 0
        df['JobRole_Research Director'] = 0
        df['JobRole_Research Scientist'] = 0
        df['JobRole_Sales Executive'] = 0
        df['JobRole_Sales Representative'] = 0
    elif JobRole == 'Manager':
        df['JobRole_Human Resources'] = 0
        df['JobRole_Laboratory Technician'] = 0
        df['JobRole_Manager'] = 1
        df['JobRole_Manufacturing Director'] = 0
        df['JobRole_Research Director'] = 0
        df['JobRole_Research Scientist'] = 0
        df['JobRole_Sales Executive'] = 0
        df['JobRole_Sales Representative'] = 0
    elif JobRole == 'Manufacturing Director':
        df['JobRole_Human Resources'] = 0
        df['JobRole_Laboratory Technician'] = 0
        df['JobRole_Manager'] = 0
        df['JobRole_Manufacturing Director'] = 1
        df['JobRole_Research Director'] = 0
        df['JobRole_Research Scientist'] = 0
        df['JobRole_Sales Executive'] = 0
        df['JobRole_Sales Representative'] = 0
    elif JobRole == 'Research Director':
        df['JobRole_Human Resources'] = 0
        df['JobRole_Laboratory Technician'] = 0
        df['JobRole_Manager'] = 0
        df['JobRole_Manufacturing Director'] = 0
        df['JobRole_Research Director'] = 1
        df['JobRole_Research Scientist'] = 0
        df['JobRole_Sales Executive'] = 0
        df['JobRole_Sales Representative'] = 0
    elif JobRole == 'Research Scientist':
        df['JobRole_Human Resources'] = 0
        df['JobRole_Laboratory Technician'] = 0
        df['JobRole_Manager'] = 0
        df['JobRole_Manufacturing Director'] = 0
        df['JobRole_Research Director'] = 0
        df['JobRole_Research Scientist'] = 1
        df['JobRole_Sales Executive'] = 0
        df['JobRole_Sales Representative'] = 0
    elif JobRole == 'Sales Executive':
        df['JobRole_Human Resources'] = 0
        df['JobRole_Laboratory Technician'] = 0
        df['JobRole_Manager'] = 0
        df['JobRole_Manufacturing Director'] = 0
        df['JobRole_Research Director'] = 0
        df['JobRole_Research Scientist'] = 0
        df['JobRole_Sales Executive'] = 1
        df['JobRole_Sales Representative'] = 0
    elif JobRole == 'Sales Representative':
        df['JobRole_Human Resources'] = 0
        df['JobRole_Laboratory Technician'] = 0
        df['JobRole_Manager'] = 0
        df['JobRole_Manufacturing Director'] = 0
        df['JobRole_Research Director'] = 0
        df['JobRole_Research Scientist'] = 0
        df['JobRole_Sales Executive'] = 0
        df['JobRole_Sales Representative'] = 1
    else:
        df['JobRole_Human Resources'] = 0
        df['JobRole_Laboratory Technician'] = 0
        df['JobRole_Manager'] = 0
        df['JobRole_Manufacturing Director'] = 0
        df['JobRole_Research Director'] = 0
        df['JobRole_Research Scientist'] = 0
        df['JobRole_Sales Executive'] = 0
        df['JobRole_Sales Representative'] = 0
    df.drop('JobRole',axis=1,inplace=True)
    prediction = model.predict (df)
    if prediction == 0:
        return render_template ('index.html',prediction_text='Employee Might Not Leave The Job')
    else:
        return render_template ('index.html',prediction_text='Employee Might Leave The Job')
        

if __name__=="__main__":
    app.run(debug=True)
    
