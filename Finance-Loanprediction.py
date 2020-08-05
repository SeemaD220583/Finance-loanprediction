#!/usr/bin/env python
# coding: utf-8

# # Loan Repayment Prediction:

# In[1]:


#Importing Libraries
import pandas as pd
import matplotlib.pyplot as  plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# # 1.Data Gathering

# In[2]:


#Load the Data
train=pd.read_csv('loan.csv')


# In[3]:


train.shape


# In[4]:


train.info()


# In[5]:


train.head(10)


# In[6]:


train.columns


# In[7]:


#Categorical Features
cat_features=train.select_dtypes(include='object').columns
print("Categorical Features:\n",cat_features,"\n No of categorical features:",len(cat_features))

#Numerical Features
num_features=train.select_dtypes(exclude='object').columns
print("Numerical Features:\n",num_features,"\n No of numerical features:",len(num_features))


# In[8]:


#Statistical analysis of numeric features
train.describe().T


# # 2. Exploratory Data Anaysis
# 

# In[9]:


#Create copy of dataframe 
df=train.copy()
df.dropna(inplace=True)
#sns.pairplot(df)


# In[10]:


#Count of target Feature
print(train['bad_loan'].value_counts())

plt.figure(figsize=(5,5))
plt.title("Count of bad_loan")
sns.countplot(train['bad_loan'],palette='plasma')
plt.xticks(ticks=np.arange(2),labels=['Good Customer','Bad Customer'])
plt.show()

plt.figure(figsize=(5,5))
train['bad_loan'].value_counts().plot(kind='pie',labels=['Good Customers','Bad Customers'],autopct="%1.2f%%")
plt.show()
#Conclusion: Dataset is imbalanced as bad_customers are less than good customers


# In[11]:


#Distribution of loan amount
plt.figure(figsize=(5,5))
plt.title("Histogram for Loan Amount ")
plt.hist(df['loan_amnt'])
plt.xlabel("Loan Amount")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(5,5))
sns.distplot(df['loan_amnt'])

#Conclusion:Major customers took loan in range of(5000-10000), few customers took loan with amoun as 30000-35000


# In[12]:


#Countplot of term 
print(train['term'].value_counts())

plt.figure(figsize=(8,5))
plt.title("Countplot for term")
sns.countplot(train['term'],palette='plasma')
plt.xticks(ticks=np.arange(2),labels=['36 Months','60 Months'])
plt.show()

plt.figure(figsize=(5,5))
train['term'].value_counts().plot(kind='pie',labels=['36 Months','60 Months'],autopct="%1.2f%%")
plt.show()
#Conclusion: Majority of loans are of 36 Month duration


# In[13]:


#Countplot for emp_length Feature
print("EmpLength\tNo of customers")
print(train['emp_length'].value_counts())

plt.figure(figsize=(5,5))
plt.title("Countplot for Employee Length")
sns.countplot(train['emp_length'],palette='plasma')
plt.xticks(ticks=np.arange(11))
plt.show()


#Conclusion: Majority of customers having employee length as 10


# In[14]:


#Relationship between Loan Amount and Annual Income
plt.figure(figsize=(5,5))
plt.scatter(df['loan_amnt'],np.log(df['annual_inc']))
plt.title("Annual Income vs Loan Amount")
plt.ylabel("Loan Amount")
plt.xlabel("Annual Income")

#Distribution of Annual Income
plt.figure(figsize=(5,5))
plt.title("Distribution Annual Income")
plt.hist(df['annual_inc'],bins=20)
plt.xticks(rotation=45)
plt.xlabel("Annual Income")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(5,5))
sns.distplot(df['annual_inc'],bins=20)

#As distribution is skewed apply log transformation on it
#Distribution of Annual Income
plt.figure(figsize=(5,5))
plt.title("Distribution Annual Income")
plt.hist(np.log(df['annual_inc']),bins=20)
plt.xticks(rotation=45)
plt.xlabel("Annual Income")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(5,5))
sns.distplot(np.log(df['annual_inc']),bins=20)

#Conclusion:Major customers took loan in range of(5000-10000), few customers took loan with amoun as 30000-35000


# In[15]:


#Count of loan purpose
print(train['purpose'].value_counts())

plt.figure(figsize=(15,5))
plt.title("Countplot for Loan Purpose")
sns.countplot(train['purpose'],palette='plasma')
plt.xlabel("Loan Purpose")
plt.xticks(rotation=45)
plt.show()


#Conclusion: Majority of customers take loan for debit consolidation followed for credit card


# In[16]:


#Statewise customer count
print(train['addr_state'].value_counts())

plt.figure(figsize=(15,5))
plt.title("Statewise Customer Count")
sns.countplot(train['addr_state'],palette='plasma')
plt.xlabel("State")
plt.ylabel("Customer Count")
plt.xticks(rotation=45)
plt.show()

#Conclusion:  state CA has majority  of customers , most of the states have average no of customers


# In[17]:


#Distribution of Debit to Income Ratio
plt.figure(figsize=(5,5))
plt.title("Distribution Debit-Income")
sns.distplot(df['dti'])
plt.xlabel("Debit-Income Ratio")
plt.ylabel("Count")
plt.show()

#Conclusion:


# In[18]:


#Distribution of Revol_util
plt.figure(figsize=(5,5))
plt.title("Distribution Revol_Util")
sns.distplot(df['revol_util'])
plt.xlabel("Revol_Util")
plt.ylabel("Count")
plt.show()

#Conclusion:


# In[19]:


#Distribution of total account 
plt.figure(figsize=(5,5))
plt.title("Histogram for total account")
plt.hist(df['total_acc'])
plt.xlabel("Total Account")
plt.ylabel("Count")
plt.show()

#Conclusion: Majority of customers have total account in range(20-40) & Distribution is skewed


# In[20]:


#Count of Longest Credit length

plt.figure(figsize=(5,5))
plt.title("Countplot for Longest Credit Length")
sns.distplot(df['longest_credit_length'])
plt.show()


#Conclusion: Distribution of credit length is skewed


# In[21]:


#Count of House Ownership
print(train['home_ownership'].value_counts())

plt.figure(figsize=(5,5))
plt.title("Countplot of House Ownership")
sns.countplot(train['home_ownership'])
plt.xticks(ticks=np.arange(6),labels=['MORTGAGE','RENT','OWN','OTHER','NONE','ANY'])
plt.show()

plt.figure(figsize=(7,5))
train['home_ownership'].value_counts().plot(kind='pie',labels=['MORTGAGE','RENT','OWN','OTHER','NONE','ANY'],autopct="%1.2f%%")
plt.show()
#Conclusion: 


# In[22]:


#Count of Verified and Not verified Customers
print(train['verification_status'].value_counts())

plt.figure(figsize=(5,5))
plt.title("Countplot by Verification Status")
sns.countplot(train['verification_status'],palette='plasma')
plt.xticks(ticks=np.arange(2),labels=['Verified','Not Verified'])
plt.show()

plt.figure(figsize=(5,5))
train['verification_status'].value_counts().plot(kind='pie',labels=['Verified','Not Verified'],autopct="%1.2f%%")
plt.show()
#Conclusion: 


# In[23]:


df.columns


# In[24]:


#Correlation of features
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,cbar=True,cmap='viridis',linewidth=0.2)
plt.yticks(rotation=0)


# In[25]:


#Checking for outlier
plt.figure(figsize=(5,5))
sns.boxplot(y=np.log(df['annual_inc']))
#There exist extrem value for annual income


# In[26]:


#Checking for outlier
plt.figure(figsize=(5,5))
sns.boxplot(y='loan_amnt',data=df)
plt.yticks(ticks=range(0,35000,5000))


# # 3.Feature Engineering

# 3.1 Handling Missing values

# In[27]:


#Checking for missing values
sns.heatmap(train.isnull(),cbar=False,cmap='viridis',yticklabels=False)


# In[28]:


train.isnull().sum()


# In[29]:


features_with_missing=[feature for feature in train.columns if train[feature].isna().sum()>=1]
features_with_missing


# In[30]:


#Compute % of missing values
for feature in features_with_missing:
    print("{} : {}".format(feature,np.round(train[feature].isna()).mean()*100,2))


# 3.2 Filling Missing Values

# In[31]:


#Employee Length:Fill na values with mean
train['emp_length'].fillna(train['emp_length'].mean(skipna=True),inplace=True)


# In[32]:


#Annual Income:mean income
train['annual_inc'].fillna(train['annual_inc'].median(),inplace=True)


# In[33]:


#delinq_2 yrs
train['delinq_2yrs'].fillna(train['delinq_2yrs'].mode()[0],inplace=True)


# In[34]:


#revol_util
train['revol_util'].fillna(train['revol_util'].median(),inplace=True)


# In[35]:


#revol_util
train['total_acc'].fillna(train['total_acc'].median(),inplace=True)


# In[36]:


#revol_util
train['longest_credit_length'].fillna(train['longest_credit_length'].median(),inplace=True)


# In[37]:


train.isnull().sum()


# In[38]:


cat_features


# 3.3 Converting Categorical-Numeric features using Label Encoding

# In[39]:


# Label encoding for variables 
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for feature in cat_features:
    train[feature]=le.fit_transform(train[feature])


# In[40]:


train.head(10)


# In[41]:


#Apply log transformation to deal with skewness of annual income
train['annual_inc']=np.log(train['annual_inc'])


# In[42]:


train.head()


# # 4. Classification Model

# # Logistic Regression

# In[43]:


# Split into train and test data
x=train.drop('bad_loan',axis=1)
y=train.bad_loan
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

#Feature Scaling
rs=RobustScaler()
x_train=rs.fit_transform(x_train)
x_test=rs.transform(x_test)


# In[44]:


#Without CV & without Feature Selection
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver='liblinear')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))


# In[45]:


#With CV & without Feature Selection
#Cross Validation
sc=RobustScaler()
x=sc.fit_transform(x)
cv_score=cross_val_score(model,x,y,cv=10)
print("Accuracy with Cross Validation:",np.mean(cv_score))


# # Feature Selection:Pearson's Correlation Coefficient

# In[ ]:


#Feature Selection : Correlation matrix
plt.figure(figsize=(15,5))
sns.heatmap(train.corr(),annot=True,cmap='viridis')


# In[47]:


#Feature Selection using correlation coefficient
correlated_features = set()
irrelevant_features=set()
correlation_matrix = train.corr()
'''
#Checking entire correlation matrix
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
        if abs(correlation_matrix.iloc[i, j]) < 0.05:
            colname = correlation_matrix.columns[i]
            irrelevant_features.add(colname)
'''
#Checking Correlation of all features with respect to target feature
i=12   #index of target feature
for j in range(0,15):
    if abs(correlation_matrix.iloc[i, j]) > 0.8:
        colname = correlation_matrix.columns[j]
        correlated_features.add(colname)
    if abs(correlation_matrix.iloc[i, j]) < 0.05:
        colname1 = correlation_matrix.columns[j]
        irrelevant_features.add(colname1)            
print("Correlated features are as: {} ,Total No of Correlated features:{}".format(correlated_features,len(correlated_features)))
print("Irrelevant features are as: {} ,Total No of Irrelevant features:{}".format(irrelevant_features,len(irrelevant_features)))

#Creates a copy of dataset
df=train.copy()
#Dropping irrelevant features
df.drop(irrelevant_features,axis=1,inplace=True)
print(df.shape)


# In[48]:


#Splitting of dataset - train,test data after dropping irrelevant features
x1=df.drop('bad_loan',axis=1)
y1=df.bad_loan
x_train1,x_test1,y_train1,y_test1=train_test_split(x1,y1,test_size=0.2,random_state=0)
print(x_train1.shape,y_train1.shape,x_test1.shape,y_test1.shape)

#Feature Scaling
rs=RobustScaler()
x_train1=rs.fit_transform(x_train1)
x_test1=rs.transform(x_test1)


# In[49]:


#Building model : Logistic Regression
#Without CV & with Feature Selection
model.fit(x_train1,y_train1)
y_pred=model.predict(x_test1)
print("Accuracy:",accuracy_score(y_test1,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test1,y_pred))
print("Confusion Matrix:\n",classification_report(y_test1,y_pred))


# In[50]:


#Cross Validation
#With CV & With Feature Selection
x1=sc.fit_transform(x1)
cv_score=cross_val_score(model,x1,y1,cv=10)
print("Accuracy with Cross Validation:",np.mean(cv_score))


# # Decision Tree

# In[51]:


#Without CV & without Feature Selection
from sklearn import tree
dtree = tree.DecisionTreeClassifier(random_state=17)
dtree.fit(x_train,y_train)
y_pred=dtree.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[52]:


#Without CV & with Feature Selection
cv_score=cross_val_score(dtree,x,y,cv=10)
print("Accuracy:",np.mean(cv_score))


# In[53]:


#Without CV & with Feature Selection
dtree.fit(x_train1,y_train1)
y_pred=dtree.predict(x_test1)
print(accuracy_score(y_test1,y_pred))
print(confusion_matrix(y_test1,y_pred))
print(classification_report(y_test1,y_pred))


# In[54]:


#With CV & with Feature Selection
cv_score=cross_val_score(dtree,x1,y1,cv=10)
print("Accuracy:",np.mean(cv_score))


# # ADABoost

# In[55]:


#Without CV & without Feature Selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
adb=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=100)
adb.fit(x_train,y_train)
y_pred=adb.predict(x_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Confusion Matrix:\n",classification_report(y_test,y_pred))


# In[60]:


#Without CV & without Feature Selection
adb=AdaBoostClassifier()
cv_score=cross_val_score(adb,x,y,cv=10)
print("Accuracy:",np.mean(cv_score))


# In[59]:


#without CV &  with Feature selection
adb.fit(x_train1,y_train1)
y_pred=adb.predict(x_test1)
print("Accuracy:",accuracy_score(y_test1,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test1,y_pred))
print("Confusion Matrix:\n",classification_report(y_test1,y_pred))


# In[61]:


#With CV & with Feature Selection
cv_score=cross_val_score(adb,x1,y1,cv=10)
print("Accuracy:",np.mean(cv_score))


# # Random Forest

# In[64]:


#Without CV & without Feature Selection
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
y_pred= rfc.predict(x_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Confusion Matrix:\n",classification_report(y_test,y_pred))


# In[65]:


#Without CV & without Feature Selection
cv_score=cross_val_score(rfc,x,y,cv=10)
print("Accuracy:",np.mean(cv_score))


# In[66]:


#Without CV & with Feature Selection
rfc.fit(x_train1, y_train1)
y_pred= rfc.predict(x_test1)
print("Accuracy:",accuracy_score(y_test1,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test1,y_pred))
print("Confusion Matrix:\n",classification_report(y_test1,y_pred))


# In[67]:


#With CV & with Feature Selection
cv_score=cross_val_score(rfc,x1,y1,cv=10)
print("Accuracy:",np.mean(cv_score))


# # SVM
#Without CV & without Feature Selection
from sklearn.svm import SVC
svc=SVC(gamma='auto')
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)

print("Support Vector Machine:")
print ("SVM:",accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))# Without CV & without Feature Selection
from sklearn.svm import SVC
cv_scores=cross_val_score(svc,x,y,cv=10)
print("Mean Accuracy:",np.mean(cv_scores))#Without CV & with Feature Selection
from sklearn.svm import SVC
svc.fit(x_train1,y_train1)
y_pred=svc.predict(x_test1)

print("Support Vector Machine:")
print ("SVM:",accuracy_score(y_test1, y_pred))
print(confusion_matrix(y_test1,y_pred))
print(classification_report(y_test1,y_pred))#With CV & with Feature Selection
from sklearn.svm import SVC
cv_scores=cross_val_score(svc,x1,y1,cv=10)
print("Mean Accuracy:",np.mean(cv_scores))
# # KNN
#Without CV & without Feature Selection
from sklearn.neighbors import KNeighborsClassifier
score=[]
for k in range(2,51):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    score.append(accuracy_score(y_pred,y_test))
plt.title("k vs Accuracy")
plt.plot(range(2,51),score,label='Accuracy')
plt.xlabel("k")
plt.ylabel("Accuracy")#with CV & without Feature Selection
#Cross Validation
model=KNeighborsClassifier(n_neighbors=)
cv_score=cross_val_score(model,x,y,cv=10)
print("Accuracy with Cross Validation:",np.mean(cv_score))#Without CV & with Feature Selection
from sklearn.neighbors import KNeighborsClassifier
score=[]
for k in range(2,51):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train1,y_train1)
    y_pred=knn.predict(x_test1)
    score.append(accuracy_score(y_pred,y_test1))
plt.title("k vs Accuracy")
plt.plot(range(2,51),score,label='Accuracy')
plt.xlabel("k")
plt.ylabel("Accuracy")

#with CV & with Feature Selection
#Cross Validation
model=KNeighborsClassifier(n_neighbors=)
cv_score=cross_val_score(model,x1,y1,cv=10)
print("Accuracy with Cross Validation:",np.mean(cv_score))