#!/usr/bin/env python
# coding: utf-8

# # PROBLEM STATEMENT

# The dataset "Sample_data" in which each row represents an independent experiment and each column represents one measurement (feature) of the experiment. The first column (ref_group) defines the experimental condition.
# A subset of these experiments belongs to the so called reference groups which are apparent by the group column:
# 
# reference group A: group == group_A
# 
# reference group B: group == group_B
# 
# reference group C: group == group_C
# 
# reference group D: group == group_D
# 
# reference group E: group == group_E
# 
# 

# # AIM

# Our target is to assign the ref_group "unknown" into any of the existing cluster groups or else we can check for the scope of determining a new group.

# # IMPORTING LIBRARIES

# In[1]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn.cluster import KMeans
import sklearn.cluster
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.model_selection import cross_validate


# # LOADING THE DATASET

# In[2]:


df =pd.read_csv(r"C:\Users\Hp\Downloads\sample_data.csv")


# In[3]:


df


# The dataset contains 10443 rows and 1490 columns. Each row represents an unique expirement result, Our target is to 
# assign the ref_group "unknown" into any of the existing cluster groups or else we can check for the scope of determining a 
# new group.

# In[4]:


#Checking the dimension of the data
df.shape


# In[5]:


#checking the datatypes of the columns in the dataset
df.dtypes


# The dataset contains "float64" and "object" datatypes

# In[6]:


df.info()


# The dataset contains 1488 numerical entries and 2 categorical columns.

# In[7]:


#View descriptive statistics of the dataset
df.describe()


# # Visualizing Null Values

# In[8]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='Blues')


# In[ ]:





# In[8]:


#Removing null values
df.dropna()


# In[9]:


#Visualization of unique values in ref_group
ref= df.ref_group.value_counts()
sns.set_style("darkgrid")
plt.figure(figsize = (10,4))
sns.barplot(x = ref.index, y=ref.values)
plt.show()


# From the above plot, 
# 
# unknow group contains 6994 values
# group_D contains 959 values
# group_E  contains 670 values
# group_B contains 576 values
# group_A contains 574 values
# group_X contains 382 values
# group_C contains 288 values
# 
# Our task is to move the features in unknown category to any of the existing Groups or check for the scope of creating a new group.

# In[10]:


df


# In[11]:


print("Total categories in the feature ref_group:\n",df["ref_group"].value_counts(), "\n")
print("Total categories in the feature_1:\n", df["feature_1"].value_counts())


# In[12]:


#The "feature_1" column contains an unique value z_score and Z_score is a scaling method, hence dropping the feature_1 from the
#dataset.
df.drop(["feature_1"], axis=1, inplace=True)


# # DATA PREPROCESSING / CLEANING

# **1.Data Cleaning**
# 

# *Visualizing outliers using Boxplot*

# In[13]:


def plot_boxplot(df,ft):
    df.boxplot(column=[ft])
    plt.grid(False)
    plt.show()


# In[14]:


plot_boxplot(df, "feature_2")


# In[15]:


plot_boxplot(df, "feature_3")


# In[16]:


plot_boxplot(df, "feature_1489")


# From the Boxplot it's clear that there are many outliers in the dataset.I It means that we need to standardize the given dataset.

# #  OUTLIERS

# In[17]:


#Detecting outliers using the descriptive statistics using interquartile ranges (IQRs)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[18]:


df


# **2.Dealing with Duplicate values**

# In[19]:


#Checking the duplicate values
df.duplicated().sum()


# There are 118 duplicate values in the given dataset. Hence, we have to drop those duplicate entries from the dataset.

# In[20]:


#Drop duplicate values from the given dataset
df = df.drop_duplicates()


# In[21]:


df


# Now we have successfully removed duplicate values from our dataset. At present our dataset contains 10325 rows × 1489 columns.

# In[22]:


#Defined a new variable "y" for storing the values in the column "ref_group" inorder to perform classification later.
y =df["ref_group"] # Created target


# In[23]:


# Inorder to keep the numeric columns for further analysis we are going to remove the categorical column "ref_group" from our dataset
df.drop(["ref_group"], axis=1, inplace=True)


# **3.Replacing Missing values with NaN**

# The given dataset contains missing values, inorder to make further analysis we need to remove those missing values. 
# We are using "replace" method to replace blank spaces with NaN. Later we use a different method for removing Null values.

# In[24]:


#Replacing blank spaces
data_new1 = df.copy()   # Create duplicate of data
data_new1 = data_new1.replace(r'^s*$', float('NaN'), regex = True)  # Replace blanks by NaN
print(data_new1)            


# In[25]:


data_new1


# In[26]:


#Checking for infinite values
count = np.isinf(data_new1).values.sum()
count


# There are 8 infinite values in the given dataset. For further analysis we need to either remove the infinite values or 
# replace it with finite values using SimpleImputer.

# # Using SimpleImputer for solving Null Values and Infinite values

# In[27]:


from sklearn.impute import SimpleImputer

data_new1.replace([np.inf, -np.inf], np.nan, inplace=True)

imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')

imputer = imp_mean.fit(data_new1)

data_new1 = imp_mean.transform(data_new1)

print(data_new1)


# In this process, we replaced infinite values with nan and the null values are replaced with the median of the respective columns. We have chosen startegy "Median" since there are many outliers in our dataset.

# # DATA ENGINEERING

# **1.Scaling**

# Units of the features are not defined, so it's best practise to standardize the data by scaling. We are using standardscaler
# from SKlearn for standardizing the numerical data.

# In[28]:


scaler = StandardScaler()
scaler.fit(data_new1)
scaled_ds = pd.DataFrame(scaler.transform(data_new1))
print("All features are now scaled")


# In[30]:


#For clustering we do need a target vector
x= scaled_ds  #numeric data
np.unique(y)#label


# **2.Principal Component Analysis**

#  Dimensionality reduction technique is used  to reduce the size of the data with high number of dimensions to 
# lesser number of dimension without the loss of information. Here we are going to apply Principal Component Analysis
# where we follow the below mentioned steps:
# 
#     create a covariance matrix 
#     
#     calculate Eigenvectors for the matrix
#     
#     Eigenvvectors are those corresponsing to large Eigen values, hence a large proportion of varianace of original data
#     is reconstructed. There might be small data loss as well. Nevertheless, the remaining eigen vectors should retains the 
#     most important variances.
# 

# In[31]:


#Create a Covariance matrix
covar_matrix = PCA(n_components = 1223 ) 


# In[32]:


#Calculating Eigenvalues
covar_matrix.fit(x)

#calculate variance ratios
variance = covar_matrix.explained_variance_ratio_


# In[33]:


var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)

#cumulative sum of variance explained with [n] features
var


# In[34]:


var[90], var[18]


# From the cumilative sum of variance (above array) we see that the first feature explains roughly 23% of the variance within our data set while the first two explain 42% and so on. Hence, it's very clear that we can obtain 90% of the variance by implementing the 18 features of the given dataset.

# In[35]:



plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(30,120)
plt.style.context('seaborn-whitegrid')


plt.plot(var)


# From the above plot, it's clear that we need to choose 18 features to get 90 % Variance and for getting 98% variance we need to 
# choose 90 features.

# In[36]:


#Define 18 features for PCA
pca=PCA(n_components=18)
reduced=pca.fit_transform(x)


# **3.KMeans For Clustering**

# We are going to train our model with reduced PCA dimension.

# In[37]:


model = sklearn.cluster.KMeans(n_clusters=6) # chosen a random number of clusters
model.fit(reduced)
y_predict = model.predict(reduced)
print(model.inertia_)
print(model.labels_)
print(model.cluster_centers_)


# In[38]:


df["cluster_id"] = model.labels_


# In[40]:


cluster_labels =model.labels_
tab = pd.crosstab(y, cluster_labels, margins=True)
tab.index = ['group_A', 'group_B', 'group_C ', 'group_D ', 'group_E', 'group_X', "group_unknown1", "total" ]
tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['Total']
tab


# # Identifying The Number Of Clusters

# In[41]:


k_list = []
elbow_scores = []

for k in range(2, 10):
    k_list.append(k)
    model = sklearn.cluster.KMeans(n_clusters=k)
    model.fit(reduced)
    es = model.inertia_
    elbow_scores.append(es)


# In[42]:


import plotly.graph_objects as go

fig = go.Figure(data=go.Scatter(x=k_list, y=elbow_scores))
fig.show()


# From the above plot it's clear that there are 6 or 7 clusters

# In[43]:


model = sklearn.cluster.KMeans(n_clusters=7)
clusters = model.fit_predict(reduced)
print(model.inertia_)
print(model.labels_)
print(model.cluster_centers_)


# In[44]:


x["label"] = clusters


# In[44]:


# PCA with 18 features
pca=PCA(n_components=18)
reduced=pca.fit_transform(x)

reduced=pd.DataFrame(np.column_stack([reduced, clusters]), columns=['feature_1','feature_2','feature_3','feature_4',
                                                                      'feature_5', 'feature_6', 'feature_7', 'feature_8',
                                                                      'feature_9', 'feature_10', 'feature_11', 'feature_12',
                                                                      'feature_13', 'feature_14', 'feature_15', 'feature_16',
                                                                       'feature_17', 'feature_18','clusters'])

sns.pairplot(reduced, hue='clusters', diag_kind=None, vars=reduced.columns[0:-1], palette='Set1')
plt.show()


# In[148]:


#PCA with 10 features
pca=PCA(n_components=10)
reduced=pca.fit_transform(x)

reduced=pd.DataFrame(np.column_stack([reduced, clusters]), columns=['feature_1','feature_2','feature_3','feature_4',
                                                                      'feature_5', 'feature_6', 'feature_7', 'feature_8',
                                                                      'feature_9', 'feature_10','clusters'])

sns.pairplot(reduced, hue='clusters', diag_kind=None, vars=reduced.columns[0:-1], palette='Set1')
plt.show()


# In[149]:


#PCA with 5 features
pca=PCA(n_components= 5)
reduced=pca.fit_transform(x)

reduced=pd.DataFrame(np.column_stack([reduced, clusters]), columns=['feature_1','feature_2','feature_3','feature_4',
                                                                      'feature_5','clusters'])

sns.pairplot(reduced, hue='clusters', diag_kind=None, vars=reduced.columns[0:-1], palette='Set1')
plt.show()


# In[45]:


#PCA with 2 features
pca=PCA(n_components=2)
reduced=pca.fit_transform(x)

reduced=pd.DataFrame(np.column_stack([reduced, clusters]), columns=['feature_1','feature_2','clusters'])

sns.pairplot(reduced, hue='clusters', diag_kind=None, vars=reduced.columns[0:-1], palette='Set1')
plt.show()


# # T_SNE Projection

# Since the dataset has more number of outliers, T_SNE can handle outliers better than PCA. Therefore, I used T_SNE to identify 
# cluster and visualize my dataset.
# 
# Reference link:
# https://www.geeksforgeeks.org/difference-between-pca-vs-t-sne/

# In[196]:



from sklearn.manifold import TSNE
tsne = TSNE(random_state=40)

X_tsne = tsne.fit_transform(x)

plt.figure(figsize=(12,10))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar()
plt.title('t_Sne projection');


# In[46]:


# Final overview after clustering analysis
cluster_labels =model.labels_
tab = pd.crosstab(y, cluster_labels, margins=True)
tab.index = ['group_A', 'group_B', 'group_C ', 'group_D ', 'group_E', 'group_X', "group_unknown1","total" ]
tab.columns = ['cluster' + str(i + 1) for i in range(7)] + ['Total']
tab


# # Conclusion for clustering

# From the above analysis,the following are my interpretations:
# 
# The given dataset forms a total of 6 clusters. Each group forms multiple clusters. Let's look at the maximum percentage of features in a group that is assigned to a single cluster; we use this simple metric to understand how groups are separated while clustering.
# 
# * In the case of Group_A, it forms 2 clusters: cluster 3 and cluster 4, the distribution of cluster is:
# cluster 3: 564
# cluster 4: 10
# then such share will be 564/574 =0.98
# 
# 
# * In the case of Group_B, it forms 1 cluster: cluster 4, the distribution of cluster is:
# cluster 4: 576
# then such share will be 576/576 = 1.0
# 
# * In the case of Group_c, it forms 2 clusters: cluster1 and cluster5. The distribution of cluster is cluster 1: 2
# cluster 5: 286
# then such share will be 280/288 = 0.99
# 
# 
# * In the case of Group_D, it forms 2 clusters: cluster1 and cluster5; the distribution of cluster is cluster 1: 543
# cluster 5: 409
# then such share will be 543/952 = 0.57
# 
# 
# * In the case of Group_E, it forms 4 clusters: cluster1 and cluster2,cluster3, and cluster5 the distribution of cluster is cluster 1: 374
# cluster 2: 1
# cluster 3 288
# cluster 5: 1
# then such share will be 374/664 = 0.56
# 
# 
# * In the case of Group_X, it forms 2 clusters: cluster1 and cluster5; the distribution of cluster is cluster 1: 188
# cluster 5: 188
# then such share will be 188/376 = 0.50
# 
# 
# * In the case of group_unknown, it forms 5 clusters: cluster 1: 4235
# cluster 2:2
# cluster 3:1
# cluster 4:2613
# cluster 5:44
# then such share will be 4235/6895 = 0.61
# 
# 
# From the results, KMeans is differentiating features very well for GroupA, GroupB, and GroupC. The clustering analysis is concluded here and now let's focus on the classification of the unknown group.

# # Classification

# In this process we need to classify the features labeled under " unknown" to any of the existing group, here we are using Support Vector Machine Algorithm for classification.

# In[47]:


#Splitting the dataset into train set and test set
from sklearn.model_selection import train_test_split
X = reduced
Y=y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,  random_state=42)


# In[48]:


#Standardizing the numeric data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# **1.SVM Classification**

# In[49]:


from sklearn.svm import SVC
#Define parameters
parameters_grid = {
    "kernel": ["rbf"], 
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "gamma": [1, 0.1, 0.01, 0.001, 0.0001]
}

#Define Grid search
model_2 = sklearn.model_selection.GridSearchCV(sklearn.svm.SVC(),
                                              parameters_grid,
                                              scoring="accuracy",
                                              cv=5, verbose=3)
#Fit the model to data
model_2.fit(X_train, Y_train)

#Generate accuracy and best hyperparameters
print("Accuracy of best SVM classfier = {:.2f}".format(model_2.best_score_))
print("Best found hyperparameters of SVM classifier = {}".format(model_2.best_params_))


# In[56]:


from sklearn.svm import LinearSVC
svc = LinearSVC(random_state=17)
svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}


# In[57]:


#Importing GridSearch for hyperparameter tunning to get best accuracy for our model
from sklearn.model_selection import GridSearchCV
best_svc = GridSearchCV(svc, svc_params, n_jobs=1, cv=3, verbose=1)
best_svc.fit(X_train_scaled, Y_train)
best_svc.predict(X_train)


# In[56]:


#Finding the best parameter and accuracy of our model
best_svc.best_params_, best_svc.best_score_


# **2.Random Forest Classifier**

# In[98]:


from sklearn.ensemble import RandomForestClassifier
parameters_grid = {
    "criterion":["gini", "entropy"],
    "n_estimators":range(50, 260, 50)
}
model_1 = sklearn.model_selection.GridSearchCV(sklearn.ensemble.RandomForestClassifier(),
                                               parameters_grid, 
                                               scoring='accuracy',
                                               cv=5,
                                               n_jobs=-1)
model_1.fit(X_train, Y_train) 
model_1.predict(X_train)

print("Accuracy of best Random Forest Classifier= {:.2f}".format(model_1.best_score_))
print("Best found hyperparameters Of Random Forest classifier= {}"
      .format(model_1.best_params_))


# **3.GradientBoostingClassifier**

# In[182]:


from sklearn.ensemble import GradientBoostingClassifier
parameters_grid = {
    "learning_rate":[0.01, 0.1, 0.2],
    "n_estimators": range(50, 260, 50),
}
model_2 = sklearn.model_selection.GridSearchCV(sklearn.ensemble.GradientBoostingClassifier(),
                                               parameters_grid, 
                                               scoring='accuracy',
                                               cv=5,
                                               n_jobs=-1)
model_2.fit(X_train, Y_train) 
model_2.predict(X_train)
print("Accuracy of best Gradient Boosting= {:.2f}".format(model_2.best_score_))
print("Best found hyperparameters Of Gradient Boosting= {}"
      .format(model_2.best_params_))


# **4.Logistic regression**

# In[191]:


from sklearn.linear_model import LogisticRegression
#create object model
model = LogisticRegression()

#define the parameters
parameters_grid = parameters_grid = [    
    {"penalty": ["l2"],"C": [100, 10, 1.0, 0.1, 0.01],
     "solver": ["lbfgs","newton-cg","liblinear"],
     "max_iter": [100, 1000,2500, 5000]
    }
]
#definining Gridsearch
model_6 = GridSearchCV(model, param_grid=parameters_grid,
                       cv=3, verbose=True, n_jobs=-1)

#Fit model to data
model_6.fit(X_train, Y_train)
model_6.predict(X_train)

#Generate accuracy and best Hperparameters 
print("Accuracy of best logistic regression classfier= {:.2f}"
     .format(model_2.best_score_))
print("Best found hyperparameters of logistic regression classifier= {}"
      .format(model_2.best_params_))


# # Testing the best Model

# In[58]:


Y_predicted = best_svc.predict(X_test_scaled)
accuracy = sklearn.metrics.accuracy_score(Y_test, Y_predicted)
Y_predicted


# # Classification Results

# In[65]:


#Using crosstab to describe how the model has classified the values in unknown category (group_unknown1)
tab = pd.crosstab(Y_test, Y_predicted, margins=True)
tab.index = ['group_A', 'group_B', 'group_C ', 'group_E', 'group_D', 'group_X', "group_unknown1", 'total' ]
tab.columns = ['group_A', 'group_B', 'group_C ', 'group_D','group_E', 'group_X' ]
tab


# As per the current results of our Pipeline, we found that the "objects" in the unknown category belong to "Group_E" and we found an additional cluster"Group_X" which doesnot belong to this category and we can ignore it. We used the SVM classification method to classify the objects in the unknown category and obtained an accuracy of 80%, which is fair enough when we consider the fact that there are many outliers in the given dataset. 
# 

# # SUMMARY

# We initially did data cleaning with the given dataset, where we removed duplicate values and replaced blank spaces with null values. We adopted the SimpleImputer technique to replace null values with the chosen "median" of the respective columns. We have selected the strategy as median since our dataset contains many outliers. 
# 
# The dataset comprises high dimensions. We applied dimensionality reductions using "Principal Component Analysis."
# From the results of PCA, we see that the first feature explains roughly 23% of the variance within our data set while the first two explain 42% and so on. If we employ 18 features, we capture 90% of the variance within the dataset. Thus we gain very little by implementing an additional feature. Hence We chose the number of components to be considered for clustering analysis as 18.
# 
# We trained our model using Kmeans Clustering on reduced PCA dimension. We used the elbow method to identify the number of clusters for the dataset. We got the number of clusters "6".
# 
# Once we had completed the clustering analysis then, we focussed on the classification of features in the "unknown" Group. We used the SVM classification algorithm for the classification/Grouping of objects in an unknown category into any of the existing groups. As per the results of SVM, objects in the unknown category belong to "Group E"  ( accuracy = 80%)

# **Unable to upload the link for the dataset as it's a unique private data**

# **REFERENCES**
#     
#     https://www.kaggle.com/code/vjchoudhary7/kmeans-clustering-in-customer-segmentation
#     https://scikit-learn.org/stable/modules/ensemble.html
#     https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
#     

# In[ ]:




