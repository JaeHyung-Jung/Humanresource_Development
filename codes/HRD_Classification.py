#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Classification Written by Jung-Jaehyung referenced pytorch_official_guide for HRD 2021.08.10~08.11
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


dataset = pd.read_excel('Data_max_80.xlsx')
dataset.head()


# In[ ]:


# x,y 에 각각 첫번째열 두번째열 할당(x = input, y = output)
X = dataset.iloc[:, 0]
y = dataset.iloc[:, 1]


# In[ ]:


trainloader = torch.utils.data.DataLoader(X, batch_size=16,
                                         shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(y, batch_size=16, shuffle=False, num_workers=2)

classes = ('1', '2', '3', '4', '5')


# In[ ]:


# x, y확인
print("X = ", X)
print("y = ", y)


# In[ ]:


# train, test split, hyper parameters 정의
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
batch_size = 16


# In[ ]:


# data iteration, dataloader를 이용해 ran
#dataiter = iter(trainloader)
#images, labels = dataiter.next()


# In[ ]:


# 신경망 정의
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 64, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, X):
        X = self.pool(F.relu(self.conv1(X)))
        X = self.pool(F.relu(self.conv2(X)))
        X = self.pool(F.relu(self.conv3(X)))
        X = self.pool(F.relu(self.conv4(X)))
        X = torch.flatten(X, 1) # flatten all dimensions except batch
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(x)
        return X

net = Net()        


# In[ ]:


# Loss function, optimizer 정의
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)


# In[ ]:


# Network train 수정필요!!
for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        X,y = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


# In[ ]:


# 예측 
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        X,y = data
        outputs = net(X)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                   accuracy))


# In[ ]:




