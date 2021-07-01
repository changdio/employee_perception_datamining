#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing numpy and pandas
import numpy as np
import pandas as pd

# Importing plot libraries
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')


# In[4]:


# Import employee survey dataset
df = pd.read_csv('public-sector-commission-eps-2016and2015.csv')
df


# In[3]:


# Agreeable level that employee is satisfied with their job.
# 1 for Strongly agree and 7 for Strongly disagree.

df['job_satisfaction_level'].value_counts()


# In[4]:


df.columns


# In[5]:


# Ratings except education_level and salary_rate originally reversed: 1 -> 7 = Best -> worst 
# Changing to 1 -> 7 as from worst to best

df2 = df[['job_satisfaction_level','level_of_skill_used', 'expectations_clarity','contribution_recognition', 
          'career_development', 'well_managed','worklife_balance', 'effective_communication', 'respectful_supervisor',
          'good_relationship_with_coworkers', 'ethical_practices',
           'conflicts_immediately_resolved']].replace({1:7,2:6,3:5,4:4,5:3,6:2,7:1})

df2.head()


# In[6]:


# Adding columns education_level and salary_rate back in
df2['education_level'] = df['education_level']
df2['salary_rate'] = df['salary_rate']

df2.head()


# In[7]:


# Checking if ratings are changed correctly
df.head()


# In[8]:


df2.job_satisfaction_level.value_counts()


# In[9]:


# Creating function to classify that the employee likes the job or not like.
def job_satisfaction(x):
    if x >= 5:
        return 1
    else:
        return 0
    
# Adding 'likes_job' column where 0 if employee does not like job and 1 meaning the like their job.
df2['likes_job'] = df2['job_satisfaction_level'].apply(job_satisfaction)

# Check table
df2


# In[10]:


df2.columns


# In[11]:


# Exploratory Data Analysis


fig2 = plt.figure(figsize=(15,25))

# Level of skills the job requires
ax1 = fig2.add_subplot(621)
ax1.set_xlabel('level_of_skill_used')
ax1.set_ylabel('count')
sns.countplot(data = df2, x = 'level_of_skill_used', hue = 'likes_job')

# Clear Expectations
ax2 = fig2.add_subplot(622)
ax2.set_xlabel('expectations_clarity')
ax2.set_ylabel('count')
sns.countplot(data = df2, x = 'expectations_clarity', hue = 'likes_job')

# How often recognized for contributions
ax3 = fig2.add_subplot(623)
ax3.set_xlabel('contribution_recognition')
ax3.set_ylabel('count')
sns.countplot(data = df2, x = 'contribution_recognition', hue = 'likes_job')

# Management level
ax4 = fig2.add_subplot(624)
ax4.set_xlabel('well_managed')
ax4.set_ylabel('count')
sns.countplot(data = df2, x = 'well_managed', hue = 'likes_job')

# Work/life balance
ax5 = fig2.add_subplot(625)
ax5.set_xlabel('worklife_balance')
ax5.set_ylabel('count')
sns.countplot(data = df2, x = 'worklife_balance', hue = 'likes_job')

# Communication between departments
ax6 = fig2.add_subplot(626)
ax6.set_xlabel('effective_communication')
ax6.set_ylabel('count')
sns.countplot(data = df2, x = 'effective_communication', hue = 'likes_job')

# Supervisor rating
ax7 = fig2.add_subplot(627)
ax7.set_xlabel('respectful_supervisor')
ax7.set_ylabel('count')
sns.countplot(data = df2, x = 'respectful_supervisor', hue = 'likes_job')

# Relationship with peers
ax8 = fig2.add_subplot(628)
ax8.set_xlabel('good_relationship_with_coworkers')
ax8.set_ylabel('count')
sns.countplot(data = df2, x = 'good_relationship_with_coworkers', hue = 'likes_job')

# Company Ethics
ax9 = fig2.add_subplot(629)
ax9.set_xlabel('ethical_practices')
ax9.set_ylabel('count')
sns.countplot(data = df2, x = 'ethical_practices', hue = 'likes_job')

# How quick are conflicts resolved
ax10 = fig2.add_subplot(6,2,10)
ax10.set_xlabel('conflicts_immediately_resolved')
ax10.set_ylabel('count')
sns.countplot(data = df2, x = 'conflicts_immediately_resolved', hue = 'likes_job')

# Education Level
ax11 = fig2.add_subplot(6,2,11)
ax11.set_xlabel('education_level')
ax11.set_ylabel('count')
sns.countplot(data = df2, x = 'education_level', hue = 'likes_job')

# Salary
ax12 = fig2.add_subplot(6,2,12)
ax12.set_xlabel('salary_rate')
ax12.set_ylabel('count')
sns.countplot(data = df2, x = 'salary_rate', hue = 'likes_job')


# In[12]:


df2['education_level'].value_counts()


# In[13]:


df2['salary_rate'].value_counts()


# In[14]:


df2.groupby('likes_job').mean()


# In[15]:


# Features. Dropping target variables.

x = df2.drop(['likes_job','job_satisfaction_level'], axis=1)
x


# In[16]:


# Multicollinearity quick check
x.corr()


# In[17]:


# Class variable

y = df2['likes_job']


# In[18]:


# Machine Learning Imports. Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# For evaluation later
from sklearn import metrics


# In[19]:


# Creating Logistic Regression Model. Not training model yet. For accuracy comparison.

log = LogisticRegression()

# Fitting data
log.fit(x,y)

# Accuracy check
log.score(x,y)


# In[20]:


# Flattened array of feature coefficients
np.ravel(log.coef_)


# In[21]:


# Coefficients from the model

coeff_df2 = pd.DataFrame(x.columns, columns = ['Features'])
coeff_df2['Coefficients'] = np.ravel(log.coef_)


# In[22]:


coeff_df2


# ## Logistic Regression - Training and Testing Data Set

# In[23]:


# Splitting the data. Test size set to 33% and arbitrary random_state = 100

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 100)


# In[24]:


# Creating Logistic Regression model
log2 = LogisticRegression()

# Fitting new model
log2.fit(X_train, y_train)


# In[25]:


# Predict the classes of the testing data set. Employee likes job or not.
y_predict = log2.predict(X_test)

# Accuracy score
metrics.accuracy_score(y_test, y_predict)


# In[26]:


# Classification report
from sklearn.metrics import classification_report


# In[27]:


print(classification_report(y_test, y_predict))


# In[28]:


# Confusion matrix

pd.crosstab(y_test,y_predict,rownames=['Actual'],colnames=['Predicted'],margins=True)


# In[29]:


# Improved confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
plt.figure(figsize = (10,7))
sns.heatmap(cm, annot = True, cmap="YlGnBu")
plt.xlabel('Predicted')
plt.ylabel('Actual')


# ## Decision Tree

# In[30]:


# Decision tree imports
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split


# In[31]:


# Train test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=100)


# In[32]:


# Decision tree model
c = DecisionTreeClassifier(min_samples_split=101)


# In[33]:


# Fit model
c.fit(X_train,y_train)


# In[34]:


# Predict classes
y_pred = c.predict(X_test)


# In[35]:


# Accuracy score
metrics.accuracy_score(y_test, y_pred)


# In[36]:


# Classification report
print(classification_report(y_test, y_pred))


# In[37]:


# Confusion matrix

pd.crosstab(y_test,y_pred,rownames=['Actual'],colnames=['Predicted'],margins=True)


# In[38]:


# Confusion matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (10,7))
sns.heatmap(cm, annot = True, cmap="YlGnBu")
plt.xlabel('Predicted')
plt.ylabel('Actual')


# ## Naive Bayes

# In[39]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 100)


# In[40]:


# Naive Bayes model

from sklearn.naive_bayes import MultinomialNB

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)


# In[41]:


# Predict testing set
predictions = naive_bayes.predict(X_test)


# In[42]:


# Accuracy score
metrics.accuracy_score(y_test, predictions)


# In[43]:


naive_bayes.predict([[5,4,6.5,7,3,4,2,1,4,5,6,5,4]])


# In[44]:


# Classification report
print(classification_report(y_test, predictions))


# In[45]:


# Confusion matrix

pd.crosstab(y_test,predictions,rownames=['Actual'],colnames=['Predicted'],margins=True)


# In[46]:


# Confusion matrix

cm = confusion_matrix(y_test, predictions)
plt.figure(figsize = (10,7))
sns.heatmap(cm, annot = True, cmap="YlGnBu")
plt.xlabel('Predicted')
plt.ylabel('Actual')


# In[47]:


# ROC Curves of all three models. AUC scores beside them.

y_pred_proba = log2.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="Logistic Regression, auc="+str(auc))

y_pred_proba2 = c.predict_proba(X_test)[::,1]
fpr2, tpr2, _ = metrics.roc_curve(y_test, y_pred_proba2)
auc2 = metrics.roc_auc_score(y_test, y_pred_proba2)
plt.plot(fpr2,tpr2,label="Decision Tree, auc="+str(auc2))

y_pred_proba3 = naive_bayes.predict_proba(X_test)[::,1]
fpr3, tpr3, _ = metrics.roc_curve(y_test, y_pred_proba3)
auc3 = metrics.roc_auc_score(y_test, y_pred_proba3)
plt.plot(fpr3,tpr3,label="Naive Bayes, auc="+str(auc3))

plt.legend(loc=4)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curves')
plt.show()


# ## Unsupervised Learning - Apriori algorithm

# In[48]:


# Apriori imports

import csv
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# In[49]:


# Removing class variables. Only comparing features.
df_a = df2.drop(['job_satisfaction_level','likes_job'],axis=1)


# In[50]:


df_a


# In[51]:


# Transforming rating (numerical variables) into categorical (strongly disagree to strongly agree)

df_a = df_a[['level_of_skill_used', 'expectations_clarity','contribution_recognition', 
          'career_development', 'well_managed','worklife_balance', 'effective_communication', 'respectful_supervisor',
          'good_relationship_with_coworkers', 'ethical_practices',
           'conflicts_immediately_resolved']].replace({1:'Strongly disagree',2:'Moderately disagree',3:'Mildly disagree',4:'Neither agree nor disagree',5:'Mildly agree',6:'Moderately agree',7:'Strongly agree'})

df_a


# In[52]:


# Converting data to make it appropriate for Apriori
df_a['level_of_skill_used'] = df_a['level_of_skill_used'].map('level_of_skill_used = {}'.format)
df_a['expectations_clarity'] = df_a['expectations_clarity'].map('expectations_clarity = {}'.format)
df_a['contribution_recognition'] = df_a['contribution_recognition'].map('contribution_recognition = {}'.format)
df_a['career_development'] = df_a['career_development'].map('career_development = {}'.format)
df_a['well_managed'] = df_a['well_managed'].map('well_managed = {}'.format)
df_a['worklife_balance'] = df_a['worklife_balance'].map('worklife_balance = {}'.format)
df_a['effective_communication'] = df_a['effective_communication'].map('effective_communication = {}'.format)
df_a['respectful_supervisor'] = df_a['respectful_supervisor'].map('respectful_supervisor = {}'.format)
df_a['good_relationship_with_coworkers'] = df_a['good_relationship_with_coworkers'].map('good_relationship_with_coworkers = {}'.format)
df_a['ethical_practices'] = df_a['ethical_practices'].map('ethical_practices = {}'.format)
df_a['conflicts_immediately_resolved'] = df_a['conflicts_immediately_resolved'].map('conflicts_immediately_resolved = {}'.format)

df_a


# In[53]:


# Adding education_level. It only ranges from 1-2 unlike other features. Replaced numbers to actual meaning for Apriori.

df_a['education_level'] = df2[['education_level']].replace({1:'Education = Primary/secondary school or technical/trade certificate or diploma',2:'Education = University qualification'})


# In[54]:


# Adding back salary_rate. Five available numbers. Replaced numbers to actual salary ranges.

df_a['salary_rate'] = df2[['salary_rate']].replace([1,2,3,4,5],['Salary = Less than $60,000','Salary = $ 60,000-70,000','Salary = $ 80,000-109,999','Salary = $ 110,000-159,999','Salary = $ 160,000 and over'])


# In[55]:


df_a


# In[56]:


df_a.to_numpy()


# In[57]:


# Transform data for the apriori algorithm
te = TransactionEncoder()
te_array = te.fit(df_a.to_numpy()).transform(df_a.to_numpy())


# In[58]:


te_array


# In[59]:


te.columns_


# In[60]:


# See transformed data
te_df = pd.DataFrame(te_array, columns = te.columns_)

te_df


# In[61]:


# Expanding table for clear view of item sets and association rules.

pd.set_option('display.max_colwidth', 1)


# In[62]:


# Set minimum support threshold and show itemsets. Too many rules. Curse of dimensionality. 

freq_items = apriori(te_df, min_support = 0.03, use_colnames = True)
freq_items


# In[63]:


# If, then rules. Also setting minimum confidence thresholds

rules = association_rules(freq_items,metric='confidence',min_threshold=0.9)
rules[['antecedents','consequents','support','confidence','lift']]


# In[64]:


# Default value of display.max_rows is 10 i.e. at max 10 rows will be printed.
# Set it None to display all rows in the dataframe
pd.set_option('display.max_rows', 100)


# In[65]:


rules.head(100)


# In[ ]:




