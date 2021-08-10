#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Multi Linear Regression Written by Jung-Jaehyung referenced github(MLR) for HRD 2021.08.10

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


dataset = pd.read_excel('Data rev.3.xlsx')
dataset.head()


# In[3]:


# x,y 에 각각 첫번째열 두번째열 할당(x = input, y = output)
X = dataset.iloc[:, 0]
y = dataset.iloc[:, 1]


# In[4]:


# x, y확인
print("X = ", X)
print("y = ", y)


# In[5]:


# MLR모델에 맞게 reshape
X = X.values.reshape(-1, 1)


# In[6]:


# train, test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[7]:


# Train set을 Regression 모델에 fit하기
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[8]:


# Test set에대한 결과 예측
y_train_pred = regressor.predict(X_train)
y_pred = regressor.predict(X_test)


# In[9]:


# Y의 정답과(test) Y의 예측값 계산해야 score로 저장
from sklearn.metrics import r2_score
train_score=r2_score(y_train,y_train_pred)
test_score=r2_score(y_test,y_pred)


# In[10]:


# Y의 예측값 확인
print(y_pred)


# In[11]:


# score 계산
print("train_score : ", train_score)
print("test_score : ", test_score)


# In[12]:


# Training set visualize Results
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, y_train_pred, color = 'blue')
plt.title('Training set')
plt.xlabel('Emission')
plt.ylabel('Type')
plt.show()


# In[13]:


# Test set visualize Results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Test set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




