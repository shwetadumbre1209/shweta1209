#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[4]:


#import data
df = pd.read_csv(r'C:\Users\Admin\Desktop\freshdataset2.csv')
print(df)


# In[5]:


df.head(111)


# In[30]:


#for experiment double task CB7
df.iloc[[2,9,16,53,59,66,73,80,87,93,99,106,]]


# In[32]:


#for experiment double task CB7
df1 = df.iloc[[2,9,16,53,59,66,73,80,87,93,99,106,]]


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


# In[ ]:





# In[36]:


#To assign labels to each row
df1["Clus_km"] = labels
df1.head(12)


# In[37]:


#the distribution of people based on step length and mean stride :
area = np.pi * ( x[:, 1])**2  
plt.scatter(x[:, 1], x[:, 2], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Step length', fontsize=18)
plt.ylabel('Stride mean', fontsize=16)

plt.show()


# In[38]:


#the distribution of people based on step time and  stride time :
area = np.pi * ( x[:, 1])**2  
plt.scatter(x[:, 3], x[:, 4], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Step time', fontsize=18)
plt.ylabel('Stride time', fontsize=16)

plt.show()


# In[39]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

ax.set_xlabel('Step Length')
ax.set_ylabel('mean stride')
ax.set_zlabel('Step time')

ax.scatter(x[:, 1], x[:, 2], x[:, 3], c= labels.astype(np.float))


# In[40]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

ax.set_xlabel('Step time')
ax.set_ylabel('Stride time')
ax.set_zlabel('Cadence')

ax.scatter(x[:, 3], x[:, 4], x[:, 5], c= labels.astype(np.float))


# In[ ]:





# In[16]:


#for experiment single task RG1
df2 = df.iloc[[4,11,18,24,30,36,42,48,61,68,75,82,89,95,101,108]]


# In[17]:


df2.head(16)


# In[18]:


#Standardise the data 
from sklearn.preprocessing import StandardScaler
y = df2.iloc[:, [1,2,3,4,5,6]].values
y = np.nan_to_num(y)
Clus_dataSet = StandardScaler().fit_transform(y)
Clus_dataSet


# In[19]:


#k mean clustering k=2
clusterNum = 2
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(y)
labels = k_means.labels_
print(labels)


# In[20]:


#To assign labels to each row
df2["Clus_km"] = labels
df2.head(16)


# In[21]:


# check the centroid values by averaging the features in each cluster
df2.groupby('Clus_km').mean()


# In[22]:


#the distribution of people based on step length and mean stride :
area = np.pi * ( y[:, 1])**2  
plt.scatter(y[:, 1], y[:, 2], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Step length', fontsize=18)
plt.ylabel('Stride mean', fontsize=16)

plt.show()


# In[23]:


#the distribution of people based on step time and stride time :
area = np.pi * ( y[:, 1])**2  
plt.scatter(y[:, 3], y[:, 4], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('step time', fontsize=18)
plt.ylabel('stride time', fontsize=16)

plt.show()


# In[24]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

ax.set_xlabel('mean stride')
ax.set_ylabel('step time')
ax.set_zlabel('stride time')

ax.scatter(y[:, 2], y[:, 3], y[:, 4], c= labels.astype(np.float))


# In[25]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

ax.set_xlabel('Step time')
ax.set_ylabel('Stride time')
ax.set_zlabel('Cadence')

ax.scatter(y[:, 3], y[:, 4], y[:, 5], c= labels.astype(np.float))


# In[28]:


#Standardise the data 
from sklearn.preprocessing import StandardScaler
z = df2.iloc[:, [1,2,3,4,5,6,7]].values
z = np.nan_to_num(z)
Clus_dataSet = StandardScaler().fit_transform(z)
Clus_dataSet


# In[29]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

ax.set_xlabel('Step time')
ax.set_ylabel('Stride time')
ax.set_zlabel('velocity')

ax.scatter(z[:, 3], z[:, 4], z[:, 6], c= labels.astype(np.float))


# In[ ]:




