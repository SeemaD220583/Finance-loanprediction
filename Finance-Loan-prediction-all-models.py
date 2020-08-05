#!/usr/bin/env python
# coding: utf-8

# # Loan Repayment Prediction:

# In[1]:


#Importing Libraries
import pandas as pd
import matplotlib.pyplot as  plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# # 1.Data Gathering

# In[2]:


#Load the Data & statistical analysis
def load_data():
    train=pd.read_csv('loan.csv')
    print(train.shape)
    print(train.info())
    print(train.describe().T)
    
    #Categorical Features
    cat_features=train.select_dtypes(include='object').columns
    print("Categorical Features:\n",cat_features,"\n No of categorical features:",len(cat_features))
    
    #Numerical Features
    num_features=train.select_dtypes(exclude='object').columns
    print("Numerical Features:\n",num_features,"\n No of numerical features:",len(num_features))
    return train


# # 2. Exploratory Data Anaysis
# 

# In[3]:


def exploratory_analysis(train):
    #Create copy of dataframe 
    df=train.copy()
    df.dropna(inplace=True)
    
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
    
    #Countplot for emp_length Feature
    print("EmpLength\tNo of customers")
    print(train['emp_length'].value_counts())

    plt.figure(figsize=(5,5))
    plt.title("Countplot for Employee Length")
    sns.countplot(train['emp_length'],palette='plasma')
    plt.xticks(ticks=np.arange(11))
    plt.show()
    #Conclusion: Majority of customers having employee length as 10
    
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

    #Count of loan purpose
    print(train['purpose'].value_counts())

    plt.figure(figsize=(15,5))
    plt.title("Countplot for Loan Purpose")
    sns.countplot(train['purpose'],palette='plasma')
    plt.xlabel("Loan Purpose")
    plt.xticks(rotation=45)
    plt.show()
    #Conclusion: Majority of customers take loan for debit consolidation followed for credit card
    
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
    
    #Distribution of Debit to Income Ratio
    plt.figure(figsize=(5,5))
    plt.title("Distribution Debit-Income")
    sns.distplot(df['dti'])
    plt.xlabel("Debit-Income Ratio")
    plt.ylabel("Count")
    plt.show()

    #Distribution of Revol_util
    plt.figure(figsize=(5,5))
    plt.title("Distribution Revol_Util")
    sns.distplot(df['revol_util'])
    plt.xlabel("Revol_Util")
    plt.ylabel("Count")
    plt.show()
    
    #Distribution of total account 
    plt.figure(figsize=(5,5))
    plt.title("Histogram for total account")
    plt.hist(df['total_acc'])
    plt.xlabel("Total Account")
    plt.ylabel("Count")
    plt.show()  
    
    #Count of Longest Credit length
    plt.figure(figsize=(5,5))
    plt.title("Countplot for Longest Credit Length")
    sns.distplot(df['longest_credit_length'])
    plt.show()

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

    #Correlation of features
    plt.figure(figsize=(10,10))
    sns.heatmap(df.corr(),annot=True,cbar=True,cmap='viridis',linewidth=0.2)
    plt.yticks(rotation=0)   
    
   #Checking for outlier
    plt.figure(figsize=(5,5))
    sns.boxplot(y=np.log(df['annual_inc']))
    #There exist extrem value for annual income 
    
    


# # 3.Feature Engineering

# In[4]:


def feature_engg(train):
    #Filling Missing Values
    print(train.isnull().sum())
    #Employee Length:Fill na values with mean
    train['emp_length'].fillna(train['emp_length'].mean(skipna=True),inplace=True)
    
    #Annual Income:mean income
    train['annual_inc'].fillna(train['annual_inc'].median(),inplace=True)
    
    #delinq_2 yrs
    train['delinq_2yrs'].fillna(train['delinq_2yrs'].mean(skipna=True),inplace=True)
    
    #revol_util
    train['revol_util'].fillna(train['revol_util'].median(),inplace=True)
    
    #revol_util
    train['total_acc'].fillna(train['total_acc'].median(),inplace=True)
    
    #longest_credit_length
    train['longest_credit_length'].fillna(train['longest_credit_length'].median(),inplace=True)
    print(train.isnull().sum())
    
    #Categorical Features
    cat_features=train.select_dtypes(include='object').columns
    print("Categorical Features:\n",cat_features,"\n No of categorical features:",len(cat_features))
    #Converting Categorical-Numeric features using Label Encoding
    le=LabelEncoder()
    for feature in cat_features:
        train[feature]=le.fit_transform(train[feature])
    
    #Apply log transformation to deal with skewness of annual income
    train['annual_inc']=np.log(train['annual_inc'])
    return train


# In[5]:


#Function call
train=load_data()
exploratory_analysis(train)


# In[6]:


train=feature_engg(train)
print(train.head(5))


# # 4. Classification Model

# # Logistic Regression

# In[7]:


# Split into train and test data
x=train.drop('bad_loan',axis=1)
y=train.bad_loan
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

#Feature Scaling
rs=RobustScaler()
x_train=rs.fit_transform(x_train)
x_test=rs.transform(x_test)


# In[8]:


#Without CV & without Feature Selection
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver='liblinear')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))


# In[9]:


#With CV & without Feature Selection
#Cross Validation
sc=RobustScaler()
x=sc.fit_transform(x)
cv_score=cross_val_score(model,x,y,cv=10)
print("Accuracy with Cross Validation:",np.mean(cv_score))


# # Feature Selection:Pearson's Correlation Coefficient

# In[10]:


#Feature Selection : Correlation matrix
plt.figure(figsize=(15,10))
sns.heatmap(train.corr(),annot=True,cmap='viridis')


# In[11]:


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


# In[12]:


#Splitting of dataset - train,test data after dropping irrelevant features
x1=df.drop('bad_loan',axis=1)
y1=df.bad_loan
x_train1,x_test1,y_train1,y_test1=train_test_split(x1,y1,test_size=0.2,random_state=0)
print(x_train1.shape,y_train1.shape,x_test1.shape,y_test1.shape)

#Feature Scaling
rs=RobustScaler()
x_train1=rs.fit_transform(x_train1)
x_test1=rs.transform(x_test1)


# In[13]:


#Building model : Logistic Regression
#Without CV & with Feature Selection
model.fit(x_train1,y_train1)
y_pred=model.predict(x_test1)
print("Accuracy:",accuracy_score(y_test1,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test1,y_pred))
print("Confusion Matrix:\n",classification_report(y_test1,y_pred))


# In[14]:


#Cross Validation
#With CV & With Feature Selection
x1=sc.fit_transform(x1)
cv_score=cross_val_score(model,x1,y1,cv=10)
print("Accuracy with Cross Validation:",np.mean(cv_score))


# # Decision Tree

# In[15]:


#Without CV & without Feature Selection
from sklearn import tree
dtree = tree.DecisionTreeClassifier(random_state=17)
dtree.fit(x_train,y_train)
y_pred=dtree.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[16]:


#Without CV & with Feature Selection
cv_score=cross_val_score(dtree,x,y,cv=10)
print("Accuracy:",np.mean(cv_score))


# In[17]:


#Without CV & with Feature Selection
dtree.fit(x_train1,y_train1)
y_pred=dtree.predict(x_test1)
print(accuracy_score(y_test1,y_pred))
print(confusion_matrix(y_test1,y_pred))
print(classification_report(y_test1,y_pred))


# In[18]:


#With CV & with Feature Selection
cv_score=cross_val_score(dtree,x1,y1,cv=10)
print("Accuracy:",np.mean(cv_score))


# # ADABoost

# In[19]:


#Without CV & without Feature Selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
adb=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=100)
adb.fit(x_train,y_train)
y_pred=adb.predict(x_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Confusion Matrix:\n",classification_report(y_test,y_pred))


# In[20]:


#Without CV & without Feature Selection
adb=AdaBoostClassifier()
cv_score=cross_val_score(adb,x,y,cv=10)
print("Accuracy:",np.mean(cv_score))


# In[21]:


#without CV &  with Feature selection
adb.fit(x_train1,y_train1)
y_pred=adb.predict(x_test1)
print("Accuracy:",accuracy_score(y_test1,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test1,y_pred))
print("Confusion Matrix:\n",classification_report(y_test1,y_pred))


# In[22]:


#With CV & with Feature Selection
cv_score=cross_val_score(adb,x1,y1,cv=10)
print("Accuracy:",np.mean(cv_score))


# # Random Forest

# In[23]:


#Without CV & without Feature Selection
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
y_pred= rfc.predict(x_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Confusion Matrix:\n",classification_report(y_test,y_pred))


# In[24]:


#Without CV & without Feature Selection
cv_score=cross_val_score(rfc,x,y,cv=10)
print("Accuracy:",np.mean(cv_score))


# In[25]:


#Without CV & with Feature Selection
rfc.fit(x_train1, y_train1)
y_pred= rfc.predict(x_test1)
print("Accuracy:",accuracy_score(y_test1,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test1,y_pred))
print("Confusion Matrix:\n",classification_report(y_test1,y_pred))


# In[ ]:


#With CV & with Feature Selection
cv_score=cross_val_score(rfc,x1,y1,cv=10)
print("Accuracy:",np.mean(cv_score))


# # 5. Refinement  of model :  Improving Accuracy of model
# 

# # 5.1. Removing all Categorical Features

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
df=train.drop(cat_features,axis=1)
df.columns
#Apply log transformation to deal with skewness of annual income
df['annual_inc']=np.log(df['annual_inc'])
#train-test split
# Split into train and test data
x=df.drop('bad_loan',axis=1)
y=df.bad_loan
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

#Feature Scaling
rs=StandardScaler()
x_train=rs.fit_transform(x_train)
x_test=rs.transform(x_test)


# In[ ]:


#Without CV 
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver='liblinear')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))


# # 5.2 Using PCA

# In[ ]:


from sklearn.decomposition import PCA
pca=PCA(n_components=4)
pc_x=pca.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(pc_x,y,test_size=0.3,random_state=0)

model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))

lr=LogisticRegression()
cv_score=cross_val_score(lr,pc_x,y,cv=15)
print(np.mean(cv_score))


# # 5.3 MLP Classifier

# In[ ]:


train


# In[ ]:


from sklearn.neural_network import MLPClassifier

# Split into train and test data
x=train.drop('bad_loan',axis=1)
y=train.bad_loan
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

#Feature Scaling
rs=MinMaxScaler()
x_train=rs.fit_transform(x_train)
x_test=rs.transform(x_test)

#Initialize the Multi Layer Perceptron Classifier
#model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
model=MLPClassifier()
#Train the model
model.fit(x_train,y_train)

#Predict for the test set
y_pred=model.predict(x_test)

#Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

#Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))


# # SMOTE

# In[ ]:


get_ipython().system('pip install imblearn')


# In[ ]:


import scipy


# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:



#Apply log transformation to deal with skewness of annual income
df1['annual_inc']=np.log(df1['annual_inc'])

x=df1.drop('bad_loan',axis=1)
y=df1.bad_loan

#train-test split
# Split into train and test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

#Feature Scaling
rs=StandardScaler()
x_train=rs.fit_transform(x_train)
x_test=rs.transform(x_test)
sm=SMOTE(random_state=12,ratio=1.0)


# In[ ]:




