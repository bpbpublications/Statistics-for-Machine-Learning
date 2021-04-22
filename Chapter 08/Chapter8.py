#codes are seperated by """ as multiline comments

"""
import pandas as pd
hp = pd.read_csv("hp_train.csv")
titanic = pd.read_csv("titanic_train.csv")
"""

"""
hp.head()
titanic.head()
"""

"""
import numpy as np
data_cat = data.select_dtypes(include=[object])
data_num = data.select_dtypes(include=np.number)
"""

"""
data_num.isna().sum()

cols_to_be_deleted = ["PoolQC","Fence","MiscFeature","Alley"]
data_cat.drop(cols_to_be_deleted, axis=1,inplace=True)

cols_to_replace = ["FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Electrical","MasVnrType"]
for i in cols_to_replace:
    exec("data_cat.%s.fillna(data_cat.%s.value_counts().idxmax(), inplace=True)" %(i,i))

cols_to_replace = ["LotFrontage", "GarageYrBlt", "MasVnrArea"]

for i in cols_to_replace:
    exec("data_num.%s.fillna(data_num.%s.mean(), inplace=True)" %(i,i))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data_cat_lbl = data_cat.apply(le.fit_transform)
cat = pd.DataFrame(data_cat_lbl)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
data_num_std = ss.fit_transform(data_num)
data_num_std = pd.DataFrame(data_num_std, columns=data_num.columns)

data_num_std.head()

data_final = pd.concat([data_cat_lbl, data_num_std], axis=1)

X = data_final[[“MSSubClass”]]
Y = data[[“SalePrice”]]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.1, random_state=1234)

lr = LinearRegression()
lr.fit(x_train, y_train)


pred_results =lr.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error
print("r2 score is:", r2_score(pred_results, y_test))
print("mean squared error is:", mean_squared_error(pred_results, y_test))

"""

"""
X = data_final.drop([“SalePrice”],axis=1)
Y = data[[“SalePrice”]]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.1,	random_state=1234)

#Initializing the algorithm
lr = LinearRegression()

#Fitting the algorithm
lr.fit(x_train,y_train)

#Predicting the results
pred_results =lr.predict(x_test)
result_df = pd.DataFrame(data={'SalePrice': pred_results})

#Checking the performance
print("r2 score is:", r2_score(pred_results, y_test))
print("mean squared error is:", mean_squared_error(pred_results, y_test))

"""

"""
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_deg3 = PolynomialFeatures(degree=3)

lr = LinearRegression()

X_train_poly_3 = poly_deg3.fit_transform(x_train)
X_test_poly_3 = poly_deg3.fit_transform(x_test)

lin_reg_poly_3 = lr.fit(X_train_poly_3, y_train)

pred = lin_reg_poly_3.predict(X_test_poly_3)
from sklearn.metrics import r2_score, mean_squared_error
print("r2 score is:", r2_score(pred, y_test))
print("mean squared error is:", mean_squared_error(pred, y_test))
"""

"""
#Initializing the algorithm
r = Ridge()

#Fitting the model
r.fit(x_train, y_train)

#Predicting the results
pred_results =r.predict(x_test)

#Evaluating the performance
print("r2 score is:", r2_score(pred_results2, y_test))
print("mean squared error is:", mean_squared_error(pred_results2, y_test))
"""

"""
#Initializingthealgorithm
ls=Lasso()

#Fittingthealgorithm
ls.fit(x_train,y_train)

#Predictionofresults
pred_results1=ls.predict(x_test)
result_df=pd.DataFrame(data={'SalePrice':pred_results})

#Checkingtheperformance.
print("r2scoreis:",r2_score(pred_results1,y_test))
print("meansquarederroris:",mean_squared_error(pred_results1,y_test))
"""

"""
#Initializingthealgorithm
ENet=ElasticNet()

#Fittingthealgorithm
ENet.fit(x_train,y_train)

#Predictingtheresults
pred_results=ENet.predict(x_test)
result_df=pd.DataFrame(data={'SalePrice':pred_results})

#Checkingtheperformance
fromsklearn.metricsimportr2_score,mean_squared_error
print("r2scoreis:",r2_score(pred_results,y_test))
print("meansquarederroris:",mean_squared_error(pred_results,y_test))
"""

"""
#InitializingtheAlgorithm
lr=LogisticRegression()

#FittingtheModel
lr.fit(x_train,y_train)

#Predictions
pred=lr.predict(x_test)

#CheckingthePerformance
fromsklearn.metricsimportaccuracy_score
print("Accuracyofthemodelis{0:.2f}%".format(accuracy_score(pred,y_test)*100))

"""

"""
importscipy
importmatplotlib.pyplotasplt

resid=pred_results-y_test

counts,start,dx,_=scipy.stats.cumfreq(resid,numbins=20)
x=np.arange(counts.size)*dx+start

plt.plot(x,counts,'ro')
plt.xlabel('Value')
plt.ylabel('CumulativeFrequency')

plt.show()
"""

"""
importmatplotlib.pyplotasplt
f=plt.figure(figsize=(19,15))

data_sub=data_final.iloc[:,0:8]

plt.matshow(data_sub.corr(),fignum=f.number)
plt.xticks(range(data_sub.shape[1]),data_sub.columns,fontsize=14,rotation=45)
plt.yticks(range(data_sub.shape[1]),data_sub.columns,fontsize=14)
cb=plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('CorrelationMatrix',fontsize=16)

fromstatsmodels.stats.outliers_influenceimportvariance_inflation_factor
fromstatsmodels.tools.toolsimportadd_constant

X=add_constant(data_sub)

pd.Series([variance_inflation_factor(X.values,i)
foriinrange(X.shape[1])],
index=X.columns)
"""

