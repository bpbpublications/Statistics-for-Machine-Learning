#codes are seperated by """ as multiline comments

"""
#Importing Important Library
import scipy.stats as stats

#Defining our Data
method_1 = [81,81,85,67,88,72,80,63,62,92,82,49,69,66,74,80]
method_2 = [85,53,80,75,64,39,60,61,83,66,75,66,90,93]
method_3 = [81,59,70,70,64,78,75,80,52,45,87,85,79]
method_4 = [86,90,81,61,84,72,56,68,82,98,79,74,82]

#Defining the Level of Significance
alpha = 0.05 

#Performing the One Way ANOVA Test
outcome = stats.f_oneway(method_1, method_2, method_3, method_4)

#Printing the Outcome
print("F-Statistic Value is %f and P-Value is %f" %(outcome.statistic, outcome.pvalue))

#Checking the result
if outcome.pvalue<= alpha:
print( 'Null hypothesis can be rejected \n\n P-value is less than the level of Significance \n F-statistic is greater than F-critical')
else:
print( '\nNull hypothesis cannot be rejected \n\n P-value is greater than the level of Significance \n F-statistic is less than F-critical')
"""

"""
#Importing Important Library
import scipy.stats as stats

#Defining our Data
lean = [844.2,745,773.1,823.6,812,758.9,810.7,790.6]
mixed = [897.7,908.1,948.8,836.6,871.6,945.9,859.4,920.2]
higher_fat = [843.4,862.2,790.5,876.5,790.8,847.2,772,851.3]

#Defining the Level of Significance
alpha = 0.05 

#Performing the One Way ANOVA Test
outcome = stats.f_oneway(lean, mixed, higher_fat)

#Printing the Outcome
print("F-Statistic Value is %f and P-Value is %f" %(outcome.statistic, outcome.pvalue))


#Checking the result
if outcome.pvalue<= alpha:
print( 'Null hypothesis can be rejected \n\n P-value is less than the level of Significance \n F-statistic is greater than F-critical')
else:
print( '\nNull hypothesis cannot be rejected \n\n P-value is greater than the level of Significance \n F-statistic is less than F-critical')
"""

"""
#importing necessary library
import pandas as pd

#converting our data into pandas dataframe
df = pd.DataFrame()
df['treatment1'] = lean
df['treatment2'] = mixed
df['treatment3'] = higher_fat
display(df)
"""

"""
#stacking everything into three columns – id, treatment, result
stacked_data = df.stack().reset_index()
stacked_data = stacked_data.rename(columns={'level_0': 'id',
                                            'level_1': 'treatment',
                                            0:'result'})
display(stacked_data)

"""

"""
#Applying Tukey Test
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
MultiComp = MultiComparison(stacked_data['result'], stacked_data['treatment'])
print(MultiComp.tukeyhsd().summary())

"""

"""
#Preparing Data
data = [175,193,180,192,213,222,205,226,203,185,218,231]
gender = ["Male","Male","Female","Female"]*3
patient = ['1','2']*6
age = [["18-34"]*4,["35-54"]*4,["55+"]*4]

age_flattened = []
for sublist in age:
    for item in sublist:
age_flattened.append(item)

#Creating a DataFrame
a=pd.DataFrame()
a["Cholestrol_level"] = data
a["Gender"] = gender
a["Patient_No"] = patient
a["Age"] = age_flattened
display(a)

"""

"""
#Generating the Model to check for Interaction Effect
from statsmodels.formula.api import ols
model = ols('Cholestrol_level ~ C(Gender)*C(Age)', a).fit()
print(f"Overall model F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}")
"""

"""
#Checking Residual Summary
res = sm.stats.anova_lm(model, typ= 2)
display(res)
"""

"""
#Checking the Model to check for Main Effect
model = ols('Cholestrol_level ~ C(Gender)+C(Age)', a).fit()
print(f"Overall model F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}")

"""

"""

#Checking Residual Summary
res2 = sm.stats.anova_lm(model, typ= 2)
display(res2)

"""

"""
import pandas as pd
from statsmodels.multivariate.manova import MANOVA

group1 = [12,10,16,9,11,11,12,11,6,15,9,10,11,13,14]
group2 =[6,20,11,14,5,7,14,15,13,5,4,7,9,11,12]
group3=[12,13,14,12,11,10,10,15,15,8,9,7,6,11,13]
group4=[10,9,12,5,14,5,7,9,13,10,9,10,15,12,8]

b = pd.DataFrame()
b["group1"]=group1
b["group2"]=group2
b["group3"]=group3
b["group4"]=group4
b["Score"]=["BDI","BDI","BDI","BDI","BDI","HRS","HRS","HRS","HRS","HRS","SCR","SCR","SCR","SCR","SCR"]

display(b)
"""

"""
maov = MANOVA.from_formula('group1 + group2 + group3 + group4  ~ Score', data=b)
print(maov.mv_test())
"""

