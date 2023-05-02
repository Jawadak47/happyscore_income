#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

data = pd.read_csv('happyscore_income.csv')

print (data)


# In[24]:


happy = data ['happyScore']
income = data ['avg_income']

print(happy)


# In[25]:


plt.xlabel('income')
plt.ylabel('happy score')
plt.scatter(income, happy)


# In[26]:


#sorting
data.sort_values('avg_income', inplace=True)

richest = data [data['avg_income'] > 15000]

np.mean (richest['avg_income'])


# In[27]:


#lowest income in the rich set 
richest .iloc [0]


# In[28]:


#highest income in the rich set 
richest .iloc [-1]


# In[29]:


data.sort_values('avg_income', inplace=True)

richest = data[data['avg_income'] > 1500]

plt.scatter(richest['avg_income'], richest['happyScore'])

plt.text (richest.iloc[0]['avg_income'], richest.iloc[0]['happyScore'], richest.iloc[0]['country'])
    
plt.text (richest.iloc[-1]['avg_income'], richest.iloc[-1]['happyScore'], richest.iloc[-1]['country'])


# In[30]:


income_happy = np.column_stack((income, happy))

km_res = KMeans(n_clusters=3).fit(income_happy)

clusters = km_res.cluster_centers_

plt.scatter(income, happy)
plt.scatter(clusters[:,0], clusters[:,1], s=200)


# In[ ]:





# In[ ]:





# In[ ]:




