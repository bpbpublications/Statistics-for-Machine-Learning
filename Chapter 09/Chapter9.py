#codes are seperated by """ as multiline comments

"""
import pandas as pd
data = pd.read_csv("titanic_train.csv")

data.head()

data.tail()

data.shape()
"""

"""
name = ['Sachin', 'Saurav', 'Rahul', 'Virat']
age = [39, 40, 41, 30]

zipped =  list(zip(name,age))

crkt_data = pd.Dataframe(zipped, columns = ['Name', 'Age'])
crkt_data

"""

"""
dt = {'Name' : ['Sachin', 'Saurav', 'Rahul', 'Virat'], 'Age' : [39, 40, 41, 30]}
crkt_data = pd.Dataframe(dt)
crkt_data

"""

"""
print("Printing Very First Row\n")
print(data.iloc[0])
print("\n************************************************************************************\n")
print("Printing Second Row\n")
print(data.iloc[1])
print("\n************************************************************************************\n")
print("Printing Last Row\n")
print(data.iloc[-1])
print("\n************************************************************************************\n")
print("Printing First Column\n")
print(data.iloc[:,0])
print("\n************************************************************************************\n")
print("Printing Second Column\n")
print(data.iloc[:,1])
print("\n************************************************************************************\n")
print("Printing Last Columns\n")
print(data.iloc[:,-1])
"""

"""
print("Printing 5th to 7th Columns having first 4 rows\n")
print(data.iloc[0:5, 5:8])
print("\n************************************************************************************\n")
print("Printing 1st, 4th, 7th, 25th row + 1st 6th 7th columns\n")
print(data.iloc[[0,3,6,24], [0,5,6]])

"""

"""
importnumpy as np
data_cat = data.select_dtypes(object)
data_num = data.select_dtypes(np.number)

data_cat.columns
data_num.columns
data_cat.isna().sum()
data_num.isna().sum()

data_cat['Embarked'].fillna(data_cat['Embarked'].value_counts().idxmax(), inplace=True)

data_cat.drop(['Cabin'], axis=1, inplace=True)

data_num['Age'].fillna(data_num['Age'].mean(), inplace=True)

data_cat['Embarked'].ffill(inplace=True)
data_cat['Embarked'].bfill(inplace=True)

data_num['Age'].interpolate('polynomial', order=2, inplace=True)
data_num['Age'].interpolate('linear', inplace=True)

data.dropna(axis=0, inplace=True)
"""

"""
student_id = [0,1,2,3,4,5,6,7,8,9,10]

student_name = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H','I', 'J']

student_id_sports = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

sports = ['Cricket', 'Football', 'BasketBall', 'Football','Cricket', 'Football', 'BasketBall', 'Football','Cricket', 'Football', 'BasketBall', 'Football','Cricket', 'Football', 'BasketBall', 'Football','Cricket', 'Football', 'BasketBall', 'Football','Cricket', 'Football', 'BasketBall', 'Football']

student = pd.Data frame(zip(student_id, student_name), columns=['Id', 'Name'])
sports_df = pd.Dataframe(zip(student_id_sports, sports), columns=['Id', 'Sports'])

student.merge(sports_df, how='inner', on='Id')
student.merge(sports_df, how='outer', on='Id')
student.merge(sports_df, how='left', on='Id')
student.merge(sports_df, how='right', on='Id')
"""

"""
import pandas as pd
data = pd.read_excel('titanic_train.xlsx')

data1 = pd.read_excel('titanic_train.xlsx', sheet_name='train_1')
data2 = pd.read_excel('titanic_train.xlsx', sheet_name='train_2')

# Initialize the Excel Writer Engine
writer = pd.ExcelWriter('three_sheets.xlsx', engine='xlsxwriter')

# Save Data frames to Different Sheets
df1.to_excel(writer, index=False, sheet_name='train_1')
df2.to_excel(writer, index=False, sheet_name='train_2')
df3.to_excel(writer, index=False, sheet_name='train_3')

# Add the sheets to the engine
workbook = writer.bookworksheet = writer.sheets['train_1']
workbook = writer.bookworksheet = writer.sheets['train_2']
workbook = writer.bookworksheet = writer.sheets['train_3']

# Write on the Disk
writer.save()

"""

"""
Female_Fare = data[data['Sex']=='female']['Fare']
pd.cut(Female_Fare, [6.75, 12.08, 23.1, 56, 513], labels = ['Low', 'Medium', 'High', 'Very High']).value_counts()

Male_Fare = data[data['Sex']=='male']['Fare']
pd.cut(Male_Fare, [0, 8, 11, 27, 513], labels = ['Low', 'Medium', 'High', 'Very High']).value_counts()

"""

"""
data[‘Fare’] or data.Fare
data[['Fare']]
names = ['Sachin', 'Saurav', 'Rahul']
df = pd.Series(names)
df
"""

"""
a = np.arange(10)
print(a)
print(a[5])

print(a[0:4])

a = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(a)

print(a[1])

print(a[1][0])

print(a[-1])

a = np.arange(20)
print(a)
print(a[-6:19:1])

reversed_a = a[:,:,-1]
print(reversed_a)

"""

"""
arr = np.arange(1,100)
shape10x10 = arr.reshape(10,10)
shape25x4= arr.reshape(25,4)
shape50x2= arr.reshape(50,2)

print(shape10x10)
print(shape25x4)
print(shape50x2)

"""

"""
a = np.random.randint(1,100,100)
b = np.random.normal(100)
c = np.random.uniform(100)

print(a)
print(b)
print(c)

z = np.arange(1,100)
print(np.random.shuffle(z))

"""

"""
a = np.random.randint(1,100,100)
print(np.mean(a))
print(np.median(a))
print(np.ptp(a))
print(np.var(a))
print(np.std(a))
print(np.min(a))
print(np.max(a))

"""

"""
# Generating some random arrays
a = np.random.randint(1, 100, 10).reshape(5,2)
b = np.random.normal(size=10).reshape(2,5)
c = np.random.uniform(size=10).reshape(5,2)
z = np.array([2,2,2,4,4,4,6,6,6,7])

# Checking the Shape of the array
print(a.shape, b.shape, c.shape)

# Performing Mathematical operations on above arrays
print("\n************************************************************************************\n")
print("Dot Product of arrays is: ", np.dot(a,b), "\n")
print("Multiplication result of arrays is: ", np.multiply(a,c), "\n")
print("Addition result of arrays is: ", np.add(a,c), "\n")
print("Division result of arrays is: ", np.around(np.divide(a,c),2), "\n")
print("Negative version of arrays is: ", np.negative(a,c), "\n")
print("Absolute Value of array is: ", np.abs(a), "\n")
print("Square Root of array is: ", np.sqrt(a), "\n")
print("Exponential values of array is: ", np.exp(b), "\n")

# Other important operations on arrays
print("\n************************************************************************************\n")
print("Sorting the array: ", np.sort(a), "\n")
print("Finding the floor: ", np.floor(b), "\n")
print("Finding the ceil: ", np.ceil(b), "\n")
print("Truncating the array: ", np.trunc(b), "\n")
print("Finding unique numbers: ", np.unique(z), "\n")

"""

"""
# Padding an array with zeroes.
print("Padding an array with zeroes.")
A = np.array([1,2,3,4,5])
print(np.pad(A, (2, 3), 'constant').reshape(2,5))
print("\n************************************************************************************\n")


# Adding null value to an array and checking its presence
print("Adding null value to an array and checking its presence")
B = np.array([1,2,np.nan, 4])
print(B)
print(np.isnan(B))
print("\n************************************************************************************\n")


# Checking data types of arrays
print("Checking data types of arrays")
print(a.dtype, b.dtype, c.dtype)
print("\n************************************************************************************\n")

# Conversion of one type array to another
print("Conversion of one type array to another")
print(b.astype(int))
print("\n************************************************************************************\n")

# Creating equal length stops
print("Creating equal length stops")
print(np.linspace(1,10,10))
print(np.linspace(1,10,5))
print("\n************************************************************************************\n")

# Argument based numpy methods
print("Argument based numpy methods")
t = [0, 1,4,2,3,5,6,1,9,1,6]
print(np.argmax(t))
print(np.argmin(t))
print(np.argsort(t))
print(np.sort(t))
print("\n************************************************************************************\n")

# using flat() object
print("using flat() object")
for i in a.flat:
print(i)
print("\n************************************************************************************\n")

# Finding shape of arrays
print("Finding shape of arrays")
print(a.shape, b.shape, c.shape)
print("\n************************************************************************************\n")

# Converting a list to array and reshaping it
print("Converting a list to array and reshaping it")
ls = [1,2,3,4,5,6,7,8,9,0]
print(np.asarray(ls))
print(np.asarray(ls).reshape(2,5))
print("\n************************************************************************************\n")

# Checking for a condition inside an array
print("Checking for a condition inside an array")
print(np.where(b<0, 'negative', 'positive'))

"""