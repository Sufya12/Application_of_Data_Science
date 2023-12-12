#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[6]:


dataframe = pd.read_csv('Iris.csv')
dataframe


# In[10]:


dataframe.head()


# In[11]:


dataframe.describe()


# In[12]:


print('median', dataframe['SepalLengthCm'].median())
print(dataframe['SepalLengthCm'].describe())

print('median', dataframe['SepalWidthCm'].median())
print(dataframe['SepalWidthCm'].describe())

print('median', dataframe['PetalLengthCm'].median())
print(dataframe['PetalLengthCm'].describe())

print('median', dataframe['PetalWidthCm'].median())
print(dataframe['PetalWidthCm'].describe())


# In[16]:


plt.scatter(dataframe['SepalLengthCm'], dataframe['PetalLengthCm'])
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Relation between Sepal and Petal')
plt.show()


# In[17]:


plt.hist(dataframe['Species'])
plt.xlabel('Species')
plt.ylabel('Count')
plt.title(' Values in  Species')
plt.show()


# In[23]:


dataframe.groupby('Species')['PetalLengthCm'].mean().plot.bar()
plt.xlabel('Species')
plt.ylabel('Petal Length')
plt.title(' Petal Length with Species')
plt.show()


# In[21]:


x = dataframe[['SepalLengthCm', 'PetalLengthCm', 'SepalWidthCm', 'PetalWidthCm']]
y = dataframe['Species']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,)
print( y_test)


# In[22]:


KN = KNeighborsClassifier()
KN.fit(x_train, y_train)   
result= KN.predict(x_test)    
print(result)
print((y_test))


# In[24]:


DT = DecisionTreeClassifier()
DT.fit(x_train, y_train)    
result= DT.predict(x_test)     
print(result)
print((y_test))


# In[25]:


KN1 = KN.score(x_test, y_test)
DT1 = DT.score(x_test, y_test)

model = ['DecisionTree','KNeighbour']
predict = [KN1,DT1]
plt.bar(model,predict)
plt.xlabel('Model')
plt.ylabel('Accuracy')


# In[ ]:




