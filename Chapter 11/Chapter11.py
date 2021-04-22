#codes are seperated by """ as multiline comments

"""
import pandas as pd
import numpy as np

titanic_data = pd.read_csv("titanic_train.csv")
titanic_data.head()

# Removing unnecessary columns
titanic_data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

# Making two dataframes. One having categories, other having numbers
tit_cat = titanic_data.select_dtypes([object])
tit_num = titanic_data.select_dtypes([np.number])
tit_cat.head()
tit_num.head()

# Checking Null Values
tit_cat.isna().sum()
tit_num.isna().sum()

# Filling Null Values
tit_cat.Embarked.fillna(tit_cat.Embarked.value_counts().idxmax(), inplace=True)
tit_num.Age.fillna(tit_num.Age.mean(), inplace=True)

# Converting Categorical Columns to Label Numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
tit_cat = tit_cat.apply(le.fit_transform)

# Combining categorical and numerical dataframes
titanic_processed = pd.concat([tit_cat, tit_num], axis=1)

# Getting dependent and independent features
X = titanic_processed.drop(["Survived"], axis=1)
Y = titanic_processed[[‘Survived’]]

# Getting Train and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
len(X_train), len(X_test), len(Y_train), len(Y_test)

# Initializing all the algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
adb = AdaBoostClassifier()
bg = BaggingClassifier()
nlsvm = SVC()
lsvm = LinearSVC()
knn = KNeighborsClassifier()
gnb = GaussianNB()

# Fitting the models
dt.fit(X_train, Y_train)
rf.fit(X_train, Y_train)
adb.fit(X_train, Y_train)
bg.fit(X_train, Y_train)
nlsvm.fit(X_train, Y_train)
lsvm.fit(X_train, Y_train)
knn.fit(X_train, Y_train)
gnb.fit(X_train, Y_train)

# Prediction of Test Set
pred1 = dt.predict(X_test)
pred2= rf.predict(X_test)
pred3 = adb.predict(X_test)
pred4 = bg.predict(X_test)
pred5 = nlsvm.predict(X_test)
pred6 = lsvm.predict(X_test)
pred7 = knn.predict(X_test)
pred8 = gnb.predict(X_test)

# Checking the Model Performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
algo = [‘Decision Trees’, ‘Random Forests’, ‘Boosting’, ‘Bagging’, ‘Non Linear SVM’, ‘Linear SVM’, ‘KNN’, ‘Naïve Bayes’]
for p,i in enumerate([pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8]):
	print(f“\n**********{algo[p]}***********\n”)
print("Accuracy Score is", accuracy_score(i,Y_test))
print("Precision Score is", precision_score(i,Y_test))
print("Recall Score is", recall_score(i,Y_test))
print("F1 Score is", f1_score(i,Y_test))
"""

"""
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
import numpy as np
from scipy.spatial.distance import cdist, pdist
from matplotlib import pyplot as plt

# Reading iris data and dropping target column
data = pd.read_csv('Data/iris.data', delimiter=',', header=None)
data = data.drop([4], axis=1)
display(data.head())

# Defining train and test set
data_train = data.iloc[0:100, :]
data_test = data.iloc[100:150, :]

# Let's look at the range of k values
k_range = range(1,14)

# fit k-means for different values of k
k_means_var = [KMeans(n_clusters=k).fit(data_train) for k in k_range]

# find some metrics to decide the best k
# Step1: Calculate the Centroids
centroids = [X.cluster_centers_ for X in k_means_var]
# Step2: Find Euclidean Distance
k_euclid = [cdist(data_train, cent, 'euclidean') for cent in centroids]
# Step3: Find Minimum Distance
dist = [np.min(ke,axis=1) for ke in k_euclid]
# Step4: Calculate the Sum of Squares
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(data_train)**2)/data_train.shape[0]
bss = tss - wcss

# Displaying elbow curve to find the best value
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(k_range, bss/tss*100, 'b*-')
ax.set_ylim((0,100))
plt.grid(True)
plt.xlabel('n_clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Variance Explained vs. k')
plt.show()

# Best value comes out as 2, therefore apply k-means for 2 clusters
k_means = KMeans(n_clusters=2)
k_means.fit(data_train)

# Let's check the cluster for new data
k_means.predict(data_test)

# Check the cluster efficiency using Silhouette Score
labels = k_means.labels_
print("The Silhouette Score is: ", silhouette_score(data_train, labels, metric='euclidean'))
"""

"""
# Let us now apply Hierarchical Clustering and check the Silhouette
from sklearn.cluster import AgglomerativeClustering

# Initializing Hierarchical Clustering Algorithm
clustering = AgglomerativeClustering(linkage="ward", n_clusters=2)

# Fitting on Train Set
clustering.fit(data_train)

# Predict the clusters
y = clustering.fit_predict(data_train)

print("The Silhouette Score is: ", silhouette_score(data_train, y, metric='euclidean'))

"""

"""
from sklearn.decomposition import PCA

# Define the Components Required
pca = PCA(n_components=2)

# Apply them on Data
principalComponents = pca.fit_transform(data_train)
principalDf = pd.DataFrame(data = principalComponents,  columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, pd.read_csv('Data/iris.data', delimiter=',', header=None)[[4]]], axis = 1)

# Visualize the old and new data frames
pd.read_csv('Data/iris.data', delimiter=',', header=None).head()
finalDf.head()

# Check the total Variance Explained
print(“Total Explained Variance is: “, np.sum(pca.explained_variance_ratio_))
"""