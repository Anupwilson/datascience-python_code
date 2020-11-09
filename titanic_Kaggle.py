#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


from matplotlib import pyplot as plt


# In[ ]:


titanic_train=pd.read_csv('titanic_train.csv')
titanic_test=pd.read_csv('titanic_test.csv')


# In[5]:


# to find first 10 members on board


# In[25]:


titanic_train.head(10)


# In[24]:


# to find number of rowsand columns of data


# In[ ]:


titanic_train.shape


# In[26]:


# to find bottom passengers on board


# In[27]:


titanic_train.tail(30)


# In[22]:


titanic_train['Survived'].value_counts()


# In[21]:


# bar plot for survivors


# In[13]:


plt.figure(figsize=(5,5))
plt.bar(list(titanic_train['Survived'].value_counts().keys()),list(titanic_train['Survived'].value_counts()),color=["r","g"])
plt.show()


# In[15]:


titanic_train['Pclass'].value_counts()


# In[16]:


# bar plots for Pclass


# In[30]:


plt.figure(figsize=(5,5))
plt.bar(list(titanic_train['Pclass'].value_counts().keys()),list(titanic_train['Pclass'].value_counts()),color=["blue","green","orange"])
plt.show()


# In[31]:


titanic_train['Sex'].value_counts()


# In[33]:


plt.figure(figsize=(5,5))
plt.bar(list(titanic_train['Sex'].value_counts().keys()),list(titanic_train['Sex'].value_counts()),color=["black","red"])
plt.show()


# In[35]:


plt.figure(figsize=(5,7))
plt.hist(titanic_train['Age'])
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.show()


# In[36]:


titanic_train['Survived'].isnull()


# In[37]:


sum(titanic_train['Survived'].isnull())


# In[39]:


titanic_train['Age'].isnull()


# In[40]:


sum(titanic_train['Age'].isnull())


# In[41]:


titanic_train=titanic_train.dropna()


# In[42]:


# Building a Model


# In[43]:


sum(titanic_train['Survived'].isnull())


# In[44]:


sum(titanic_train['Age'].isnull())


# In[46]:


x_train=titanic_train[['Age']]
y_train=titanic_train[['Survived']]


# In[47]:


from sklearn.tree import DecisionTreeClassifier


# In[48]:


dtc=DecisionTreeClassifier()


# In[49]:


dtc.fit(x_train,y_train)


# In[50]:


#predicting values


# In[57]:


sum(titanic_test['Age'].isnull())


# In[59]:


titanic_test=titanic_test.dropna()


# In[60]:


x_test=titanic_test[['Age']]


# In[61]:


y_pred=dtc.predict(x_test)


# In[62]:


y_pred


# In[ ]:




