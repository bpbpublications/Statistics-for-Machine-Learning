#code can me found under triple quotes. Please run the particular code after removing the quotes above and below it.

"""
sample_space = int(input("Enter number of Elements in the Sample space"))
event = int(input("Enter number of Elements in the Event Set"))
probability = event/sample_space
print(probability)
"""

"""
A = {1,2,3,4,5,6,7,8,9}
B = {1,3,5,7,9}
print("intersection of A and B is", A.intersection(B))
"""

"""
A = {1,2,3,4,5,6,7,8,9}
B = {1,3,5,7,9}
print("Union of A and B is", A.union(B))
"""

"""
A = {1,2,3,4,5,6,7,8,9}
B = {1,3,5,7,9}
print("If A is the Universal Set, Complement of B is", A.difference(B))
"""

"""
A = {1,2,3,4,5,6,7,8,9}
C = {10,11}
print("intersection of C and A is", C.intersection(A), "\n Therefore it is a Null Set")
"""

"""
A = {1,2,3,4,5,6,7,8,9}
B = {1,3,5,7,9}
print("Is B subset of A?", B.issubset(A))
print("Is A superset of B?", A.issuperset(B))
"""


"""
defcond_prob(known_prob, combined_prob=False, conditional_prob=False):
    if conditional_prob == False:
        return(combined_prob/known_prob)
    else:
        return(conditional_prob*known_prob)

#Question 1:
print("Probability of getting both the defective fuses together ", cond_prob(known_prob=2/7, conditional_prob=1/6))

#Question 2:
print("Probability of observing at least two heads", cond_prob(combined_prob = 4/8, known_prob=7/8))
"""

"""
defdep_events(event_1, event_2):
    return(event_1*event_2)

print("the probability of getting king first and then queen is", dep_events(4/52, 4/51))
"""

"""
import scipy.stats as stats
#Answer 1
x = stats.binom(n=20, p=0.85).pmf(14)
print("Probability of exact 14 flights on time is", x)
#Answer 2
y = stats.binom(n=20, p=0.85)
total_p = 0
for k in range(1, 14):
total_p += y.pmf(k)
print("Probability of less than 14 flights on time is", total_p)
#Answer 3
print("Probability of at least 14 flights on time is", 1-total_p)
#Answer 4
z = stats.binom(n=20, p=0.85)
total_p = 0
for k in range(12, 14+1):
total_p += y.pmf(k)
print("Probability of 12 to 14 flights on time is", total_p)
"""

"""
import scipy.stats  as stats
answer = 1 - (stats.geom(p=0.9).pmf(1)+stats.geom(p=0.9).pmf(2)+stats.geom(p=0.9).pmf(3))
print("The probability of landing heads after 3rd trial is: ", answer)
"""

"""
three_particles = stats.poisson(mu=4).pmf(3)
atleast_1 = (1 - stats.poisson(mu=4).pmf(0))
six_particles = stats.poisson(mu=8).pmf(6)
print("Probability of 3 particles is: ", three_particles)
print("Probability of at least 1 particle is: ", atleast_1)
print("Probability of 6 particles in 10 seconds time period is: ", six_particles)
"""

"""
import scipy.stats  as stats
answer1 = stats.norm(50000, 20000).cdf(40000)
answer2 = stats.norm(50000, 20000).cdf(65000) - stats.norm(50000, 20000).cdf(45000)
answer3 = stats.norm(50000, 20000).cdf(70000)

print("percent of people earning less than $40,000 is {:.5f}%".format(answer1*100))
print("percent of people earning between $45,000 and $65,000 is {:.5f}%".format(answer2*100))
print("percent of people earning more than $70,000 is {:.5f}%".format((1 - answer3)*100))
"""

