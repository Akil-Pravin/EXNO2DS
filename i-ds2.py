#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
df = pd.read_csv('titanic_dataset.csv')
df


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.Survived.unique()


# In[7]:


df.rename(columns={"Sex":"Gender"},inplace=True)


# In[8]:


df


# In[9]:


import seaborn as sns


# In[10]:


sns.countplot(data=df)


# In[11]:


sns.countplot(x="Survived",hue="Gender",data=df)


# In[12]:


sns.catplot(x="Survived",hue="Gender",data=df,kind="count")


# In[13]:


sns.boxplot(data=df)


# In[14]:


sns.scatterplot(data=df)


# In[15]:


sns.scatterplot(x=df['Age'],y=df['Fare'])


# In[16]:


sns.jointplot(x='Age',y='Fare',data=df,kind="kde")


# In[17]:


sns.jointplot(x='Age',y='Fare',data=df)


# In[ ]:


sns.pairplot(data=df)


# In[ ]:




