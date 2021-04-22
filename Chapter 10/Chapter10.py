#codes are seperated by """ as multiline comments

"""
# Importing important libraries
import numpy as np
fromscipy.stats import norm

# Defining the methods required
def mean(n1, n2, n):
return ((2*n1*n2)/n)+1

def variance(n1,n2,n):
returnnp.sqrt((2*n1*n2*(2*n1*n2-n))/(n**2*(n-1)))

def statistic(r, mean, variance):
    return (r-mean)/variance

# Calculating Mean and Variance
m = mean(19,26,45)
print("Mean is: ", m)

v = variance(19,26,45)
print("Variance is: ", v)
"""

"""
z_statistic = statistic(19,m,v)
print("Z-Statistic is: ", z_statistic)

# define level of significance
a = 0.05

# Getting Critical Value
value = norm.ppf(1-a)

print("Z-Critical is: ", value)

if z_statistic>value:
print("Reject Null Hypothesis")
else:
print("Accept Null Hypothesis")
"""

"""
before = [18.5,21.5,21,20,15,19.75,15.75,18,22,15,20.5]
after = [19.25,21.75,20.25,22.25,16,19.5,17,19.25,19.5,16.5,20]

fromscipy.stats import wilcoxon

t, p = wilcoxon(after, before)

print("Test Statistic is: ", t)
print("p-value is: ", p)

level_of_significance = 0.05

if p>level_of_significance:
print("Accept Null Hypothesis")
else:
print("Reject Null Hypothesis")
"""

"""
Lincoln_Calcium = [0.11, 0.41, 0.19, 0.33, 0.09, 0.33, 0.67, 0.20, 0.21, 0.20, 0.75, 0.42, 0.09, 0.22, 
                   0.19, 0.25, 0.07, 0.34, 0.30, 0.47, 0.30, 0.46]

Clarendon_Calcium = [0.06, 0.12, 0.14, 0.10,0.09, 0.29, 0.14, 0.21,0.14, 0.10, 0.12, 0.16,0.16, 0.41, 
                     0.08, 0.13,0.03, 0.08, 0.09, 0.12]

fromscipy.stats import mannwhitneyu

t, p = mannwhitneyu(Lincoln_Calcium, Clarendon_Calcium,alternative='two-sided')

print("Test Statistic is: ", t)
print("p-value is: ", p)

level_of_significance = 0.05

if p>level_of_significance:
print("Accept Null Hypothesis")
else:
print("Reject Null Hypothesis")
"""

"""
personal_income = [48285,39817,66000,43874,32219,34453,31799,33786,37780]
birth_rate = [13.9,14.1,15.1,14.1,12.1,14.5,14.3,15.8,13.1]

fromscipy.stats import spearmanr

t, p = spearmanr(personal_income, birth_rate)

print("Test Statistic is: ", t)
print("p-value is: ", p)

level_of_significance = 0.05

if p>level_of_significance:
print("Accept Null Hypothesis")
else:
print("Reject Null Hypothesis")

"""

"""
period_A = [10456,11574,12321,11661,8521,11621,11321,7706,10872,10837]
period_B = [10799,11743,11657,11608,10982,9184,12357,12251,11916,11212]
period_C = [11465,12475,12454,12193,11244,12081,11387,9055,12319,12927]
period_D = [11261,11406,11627,8706,12022,11930,11054,11431,12618,10824]

fromscipy.stats import kruskal

t, p = kruskal(period_A,period_B,period_C,period_D)

print("Test Statistic is: ", t)
print("p-value is: ", p)

level_of_significance = 0.05

if p>level_of_significance:
print("Accept Null Hypothesis")
else:
print("Reject Null Hypothesis")
"""
