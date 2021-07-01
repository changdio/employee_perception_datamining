#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing numpy and pandas
import numpy as np
import pandas as pd

# Importing plot libraries
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')


# In[2]:


# Import employee survey dataset
df = pd.read_csv('public-sector-commission-eps-2016and2015.csv')
df


# In[3]:


# Ratings except education_level and salary_rate originally reversed: 1 -> 7 = Best -> worst 
# Changing to 1 -> 7 as from worst to best

df2 = df[['job_satisfaction_level','level_of_skill_used', 'expectations_clarity','contribution_recognition', 
          'career_development', 'well_managed','worklife_balance', 'effective_communication', 'respectful_supervisor',
          'good_relationship_with_coworkers', 'ethical_practices',
           'conflicts_immediately_resolved']].replace({1:7,2:6,3:5,4:4,5:3,6:2,7:1})

df2.head()


# In[4]:


# Adding columns education_level and salary_rate back in
df2['education_level'] = df['education_level']
df2['salary_rate'] = df['salary_rate']

df2.head()


# In[5]:


# Checking if ratings are changed correctly
df.head()


# In[6]:


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


# In[7]:


# To solve problem with curse of dimensionality. Removed these features.
# Least coefficients. Did not contribute much in predicting job satisfaction.

del df2['salary_rate']
del df2['education_level']
del df2['ethical_practices']
del df2['good_relationship_with_coworkers']
del df2['respectful_supervisor']
del df2['conflicts_immediately_resolved']


# In[8]:


df2


# In[9]:


# Features. Dropping target variables.

x = df2.drop(['likes_job','job_satisfaction_level'], axis=1)
x


# In[10]:


# Columns 
x.columns


# In[11]:


# Multicollinearity quick check
x.corr()


# In[12]:


# Class variable

y = df2['likes_job']


# In[13]:


# Machine Learning Imports. Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# For evaluation later
from sklearn import metrics


# In[14]:


# Creating Logistic Regression Model. Not training model yet. For accuracy comparison.

log = LogisticRegression()

# Fitting data
log.fit(x,y)

# Accuracy check
log.score(x,y)


# In[15]:


# Coefficients from the model

coeff_df2 = pd.DataFrame(x.columns, columns = ['Features'])
coeff_df2['Coefficients'] = np.ravel(log.coef_)

coeff_df2


# ## Logistic Regression - Training and Testing Data Set

# In[16]:


# Splitting the data. Test size set to 33% and arbitrary random_state = 100

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 100)


# In[17]:


# Creating Logistic Regression model
log2 = LogisticRegression()

# Fitting new model
log2.fit(X_train, y_train)


# In[18]:


# Predict the classes of the testing data set. Employee likes job or not.
y_predict = log2.predict(X_test)

# Accuracy score
metrics.accuracy_score(y_test, y_predict)


# In[19]:


# Classification report
from sklearn.metrics import classification_report


# In[20]:


print(classification_report(y_test, y_predict))


# In[21]:


# Confusion matrix

pd.crosstab(y_test,y_predict,rownames=['Actual'],colnames=['Predicted'],margins=True)


# In[22]:


# Visualized confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
plt.figure(figsize = (10,7))
sns.heatmap(cm, annot = True, cmap="YlGnBu")
plt.xlabel('Predicted')
plt.ylabel('Actual')


# ## Decision Tree

# In[23]:


# Decision tree imports
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split


# In[24]:


# Train test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=100)


# In[25]:


# Decision tree model
c = DecisionTreeClassifier(min_samples_split=101)


# In[26]:


# Fit model
c.fit(X_train,y_train)


# In[27]:


# Predict classes
y_pred = c.predict(X_test)


# In[28]:


# Accuracy score
metrics.accuracy_score(y_test, y_pred)


# In[29]:


# Classification report
print(classification_report(y_test, y_pred))


# In[30]:


# Confusion matrix

pd.crosstab(y_test,y_pred,rownames=['Actual'],colnames=['Predicted'],margins=True)


# In[31]:


# Visualized Confusion matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (10,7))
sns.heatmap(cm, annot = True, cmap="YlGnBu")
plt.xlabel('Predicted')
plt.ylabel('Actual')


# ## Naive Bayes

# In[32]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 100)


# In[33]:


# Naive Bayes model

from sklearn.naive_bayes import MultinomialNB

# Fitting the model. MultinomialNB() used since they are categorical values instead of numerical.
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)


# In[34]:


# Predict test data set.

predictions = naive_bayes.predict(X_test)


# In[35]:


# Accuracy score
metrics.accuracy_score(y_test, predictions)


# In[36]:


# Classification report
print(classification_report(y_test, predictions))


# In[37]:


# Confusion matrix

pd.crosstab(y_test,predictions,rownames=['Actual'],colnames=['Predicted'],margins=True)


# In[38]:


# Visual Confusion matrix

cm = confusion_matrix(y_test, predictions)
plt.figure(figsize = (10,7))
sns.heatmap(cm, annot = True, cmap="YlGnBu")
plt.xlabel('Predicted')
plt.ylabel('Actual')


# In[39]:


# ROC Curves and obtaining AUC score

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


# ## Unsupervised Learning - Apriori Algorithm

# In[40]:


# Apriori imports

import csv
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# In[41]:


# Removing class variables. Only comparing features.
df_a = df2.drop(['job_satisfaction_level','likes_job'],axis=1)


# In[42]:


df_a


# In[43]:


# Transforming rating (numerical variables) into categorical (strongly disagree to strongly agree)

df_a = df_a[['level_of_skill_used', 'expectations_clarity','contribution_recognition', 
          'career_development', 'well_managed','worklife_balance', 'effective_communication']].replace({1:'Strongly disagree',2:'Moderately disagree',3:'Mildly disagree',4:'Neither agree nor disagree',5:'Mildly agree',6:'Moderately agree',7:'Strongly agree'})

df_a


# In[44]:


# Converting data to make it appropriate for Apriori
df_a['level_of_skill_used'] = df_a['level_of_skill_used'].map('level_of_skill_used = {}'.format)
df_a['expectations_clarity'] = df_a['expectations_clarity'].map('expectations_clarity = {}'.format)
df_a['contribution_recognition'] = df_a['contribution_recognition'].map('contribution_recognition = {}'.format)
df_a['career_development'] = df_a['career_development'].map('career_development = {}'.format)
df_a['well_managed'] = df_a['well_managed'].map('well_managed = {}'.format)
df_a['worklife_balance'] = df_a['worklife_balance'].map('worklife_balance = {}'.format)
df_a['effective_communication'] = df_a['effective_communication'].map('effective_communication = {}'.format)

df_a


# In[45]:


# Convert to array in order for data to be abled to be transformed

df_a.to_numpy()


# In[46]:


# Transform data for the apriori algorithm .numpy() 
te = TransactionEncoder()
te_array = te.fit(df_a.to_numpy()).transform(df_a.to_numpy())


# In[47]:


te_array


# In[48]:


te.columns_


# In[49]:


# See transformed data
te_df = pd.DataFrame(te_array, columns = te.columns_)

te_df


# In[50]:


# Expanding table for clear view of item sets and association rules.
pd.set_option('display.max_colwidth', 1)


# In[51]:


# Set minimum support threshold and show itemsets

freq_items = apriori(te_df, min_support = 0.02, use_colnames = True)
freq_items


# In[52]:


# If, then rules. Also setting minimum confidence thresholds

rules = association_rules(freq_items,metric='confidence',min_threshold=0.9)
rules[['antecedents','consequents','support','confidence','lift']]


# In[53]:


# View rules from highest lift value to lower.

rules[['antecedents','consequents','support','confidence','lift']].sort_values(by=['lift'],ascending = False)


# In[ ]:




