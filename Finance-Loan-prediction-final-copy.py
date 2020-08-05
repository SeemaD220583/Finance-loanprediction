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
from sklearn.linear_model import LogisticRegression


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


# # 4. Classification Model

# # Logistic Regression

# In[5]:


def construct_model(train):
    # Split into train and test data
    x=train.drop('bad_loan',axis=1)
    y=train.bad_loan
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
        
    #Feature Scaling
    rs=RobustScaler()
    x_train=rs.fit_transform(x_train)
    x_test=rs.transform(x_test)
    
    model=LogisticRegression(solver='liblinear')
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print("Accuracy:",accuracy_score(y_test,y_pred))
    print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
    print("Classification Report:\n",classification_report(y_test,y_pred))

    #Cross Validation
    sc=RobustScaler()
    x=sc.fit_transform(x)
    cv_score=cross_val_score(model,x,y,cv=10)
    print("Accuracy with Cross Validation:",np.mean(cv_score))


# In[6]:


#Function call: Load the dataset and Exploratory Data Analysis
train=load_data()
exploratory_analysis(train)


# In[7]:


#Function Call:Feature Engineering & model building 
train=feature_engg(train)
print(train.head(5))
construct_model(train)

