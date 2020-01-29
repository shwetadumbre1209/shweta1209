#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[2]:


#import data
df = pd.read_csv(r'C:\Users\Admin\Desktop\actualdatafile.csv')
print(df)


# In[5]:


df.head(129)


# In[3]:


#Standardise the data 
from sklearn.preprocessing import StandardScaler
x = df.iloc[:, [1,2,3,4,5,6]].values
x = np.nan_to_num(x)
Clus_dataSet = StandardScaler().fit_transform(x)
Clus_dataSet


# In[64]:


df.iloc[:, [1,2,3,4,5,6]]


# In[65]:


df.iloc[:, [0]]


# In[ ]:





# In[18]:


df.iloc[[2,9,13,19,38,45,52,58,64,70,74,77,83,89,96,103,110,117,124]]


# In[ ]:





# In[32]:


#for experiment double task CB7
df1 = df.iloc[[2,9,13,19,38,45,52,58,64,70,74,77,83,89,96,103,110,117,124]]


# In[33]:


#Standardise the data 
from sklearn.preprocessing import StandardScaler
x = df1.iloc[:, [1,2,3,4,5,6]].values
x = np.nan_to_num(x)
Clus_dataSet = StandardScaler().fit_transform(x)
Clus_dataSet


# In[34]:


#k mean clustering k=2
clusterNum = 2
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(x)
labels = k_means.labels_
print(labels)


# In[36]:


#To assign labels to each row
df1["Clus_km"] = labels
df1.head(19)


# In[37]:


# check the centroid values by averaging the features in each cluster
df1.groupby('Clus_km').mean()


# In[39]:


#the distribution of people based on step length and mean stride :
area = np.pi * ( x[:, 1])**2  
plt.scatter(x[:, 1], x[:, 2], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Step length', fontsize=18)
plt.ylabel('Stride mean', fontsize=16)

plt.show()


# In[40]:


#the distribution of people based on step time and  stride time :
area = np.pi * ( x[:, 1])**2  
plt.scatter(x[:, 3], x[:, 4], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Step time', fontsize=18)
plt.ylabel('Stride time', fontsize=16)

plt.show()


# In[ ]:





# In[42]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

ax.set_xlabel('Step Length')
ax.set_ylabel('mean stride')
ax.set_zlabel('Step time')

ax.scatter(x[:, 1], x[:, 2], x[:, 3], c= labels.astype(np.float))


# In[43]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

ax.set_xlabel('Step time')
ax.set_ylabel('Stride time')
ax.set_zlabel('Cadence')

ax.scatter(x[:, 3], x[:, 4], x[:, 5], c= labels.astype(np.float))


# In[45]:


#for experiment single task RG1
df2 = df.iloc[[4,15,21,27,33,40,47,53,66,79,85,91,98,105,112,119,126]]


# In[46]:


df2.head(17)


# In[52]:


#Standardise the data 
from sklearn.preprocessing import StandardScaler
y = df2.iloc[:, [1,2,3,4,5,6]].values
y = np.nan_to_num(y)
Clus_dataSet = StandardScaler().fit_transform(y)
Clus_dataSet


# In[53]:


#k mean clustering k=2
clusterNum = 2
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(y)
labels = k_means.labels_
print(labels)


# In[54]:


#To assign labels to each row
df2["Clus_km"] = labels
df2.head(17)


# In[55]:


# check the centroid values by averaging the features in each cluster
df2.groupby('Clus_km').mean()


# In[56]:


#the distribution of people based on step length and mean stride :
area = np.pi * ( y[:, 1])**2  
plt.scatter(y[:, 1], y[:, 2], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Step length', fontsize=18)
plt.ylabel('Stride mean', fontsize=16)

plt.show()


# In[59]:


#the distribution of people based on step time and stride time :
area = np.pi * ( y[:, 1])**2  
plt.scatter(y[:, 3], y[:, 4], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('step time', fontsize=18)
plt.ylabel('stride time', fontsize=16)

plt.show()


# In[68]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

ax.set_xlabel('mean stride')
ax.set_ylabel('step time')
ax.set_zlabel('stride time')

ax.scatter(x[:, 2], x[:, 3], x[:, 4], c= labels.astype(np.float))


# In[ ]:




