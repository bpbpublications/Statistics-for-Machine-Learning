#codes are seperated by """ as multiline comments

"""
import numpy as np
defsample_z_score(mean_old_sample, mean_sample,std, size):
tmp = (mean_old_sample-mean_sample)/(std/np.sqrt(size))
    return tmp
answer = sample_z_score(120,110,20,25)
"""

"""
import numpy as np
defstandard_error(std, size):
tmp = std/np.sqrt(size)
    return tmp
answer = standard_error(0.284,10)
"""

"""
import numpy as np
import scipy.stats as stats
import random

#Generating Sample Population Ages have to mean like35 and 150000 data points. Package scipy is used for dummy data generation. We have passed the mean, the size, and the shift provided to the distribution.
population_ages1 = stats.poisson.rvs(loc=18, mu=35, size=150000) 

#Generating Sample Population Ages having mean as 10 and 100000 data points
population_ages2 = stats.poisson.rvs(loc=18, mu=10, size=100000)

# Concatenating the two samples to generate one combined sample
population_ages = np.concatenate((population_ages1, population_ages2))

#Let’s take 500 sample ages from the above population decided
sample_ages = np.random.choice(a= population_ages, size=500)

print("The Sample Mean is", sample_ages.mean())
print("The Population Mean is", population_ages.mean())
print("The Difference between both the means is: ", population_ages.mean() - sample_ages.mean())

std = np.std(sample_ages)
se = standard_error(std, len(sample_ages))
print("Standard Error for the Sample is", se)
print("************************")
print("%f percentage of the mean is the Standard Error, therefore it is quite precise" %((se)/sample_ages.mean()))
"""

