#codes are seperated by """ as multiline comments

"""
import numpy as np
import panda sas pd
import scipy
from scipy import stats
mu=85
x_=55
se=6.5
#CalculatingtheZvaluetocompletetheztesting
z_stat=(x_-mu)/(se/np.sqrt(25))
#calculatingthep-value
p_val=2*(1-stats.norm.cdf(z_stat))
print('Z=',z_stat,"pValue=",p_val)

"""

"""
importnumpyasnp
importpandasaspd
importscipy
fromscipyimportstats

x=[30,28,32,26,33,25,28,30]
mu=31
t_critical=2.093#foralphalevel0.05
x_=np.array(x).mean()
#subtract1fromNtogetunbiasedestimateofsamplestandarddeviation
N=len(x)

t_stat=(x_-mu)*np.sqrt(N)/np.array(x).std(ddof=1)
print("t-statistic:",t_stat)

#aonesamplet-testthatgivesyouthep-valuetoocanbedonewithscipyasfollows:
t,p=stats.ttest_1samp(x,mu)
print("t=",t,",p=",p)
"""

"""
importnumpyasnp
importpandasaspd
importscipy
fromscipyimportstats

t_critical=1.677#foralpha=0.05

dof_x1=7
dof_x2=7

dof=dof_x1+dof_x2

std_x1=1.2
std_x2=0.9
x1_=10.2
x2_=11.8

SE=np.sqrt(((dof_x1*std_x1**2+dof_x2*std_x2**2)/dof))

t_stat=(x2_-x1_)/var*np.sqrt(1/len(x1)+1/len(x2))
print("t-statistic",t_stat)

t,p=stats.ttest_ind(x2,x1,equal_var=True)
print("t=",t,",pvalue=",p)
"""

"""
importnumpyasnp
importpandasaspd
importscipy
fromscipyimportstats

x1=[39,45,21,11,38,36]#Examinationmarksbeforevacations
x2=[22,21,13,13,49,20]#Examinationmarksaftervacations
t,p=stats.ttest_rel(x2,x1)
print("t=",t,",pvalue=",p)
"""

"""
importnumpyasnp
importpandasaspd
importscipy
fromscipyimportstats

a=['rigged',3,4]
b=['fair',2,5]

Observed_Values=np.append(a[1:3],b[1:3])
print(Observed_Values)


Expected_Values=np.outer(Observed_Values,1.5)#.T[0]
print(Expected_Values)
df=1
print("DegreeofFreedom:-",df)

alpha=0.05

chi_square=sum([(o_v-e_v)**2./e_vforo_v,e_vinzip(Observed_Values,Expected_Values)])
	
#chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-squarestatistic:-",chi_square)

#implementingchisuqaretestusingscipymodule
chi_square_stat=stats.chisquare(Observed_Values,[4.5,6,3,7.5])
print("ChiSquareStatistic=",chi_square_stat[0])

#critical-value
critical_value=scipy.stats.chi2.ppf(q=1-alpha,df=df)

#p-value
p_value=1-scipy.stats.chi2.cdf(x=chi_square_stat,df=df)

print('Criticalvalue=',critical_value,'Pvalue=',p_value)
"""

"""
a=['rigged',2,3,5]
b=['fair',4,4,8]
c=['Sum',6,7,13]
df=pd.DataFrame([a,b,c])

df.columns=['dice','2','3','Sum']
df.set_index('dice',inplace=True)
obs=np.array([df.iloc[0][0:2].values,df.iloc[1][0:2].values])


observed_values=np.append(df.iloc[0][0:2].values,
df.iloc[1][0:2].values)


print("DegreesofFreedom:",2)

exp=scipy.stats.chi2_contingency(obs)[3]
exp_values=np.append(exp[0],exp[1])

chi_squared_statistic=((observed_values-exp_values)**2/exp_values).sum()
print('Chi-squaredStatistic=',chi_squared_statistic)

#implementingchisquaretestusingscipymodule
chi_stat,p_val,dof=scipy.stats.chi2_contingency(obs)[0:3]
print('Chi_square_stat',chi_stat,"p-Value",p_val,"DegreeofFreedom",dof)
"""