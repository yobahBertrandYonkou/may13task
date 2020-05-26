#!/usr/bin/env python
# coding: utf-8

# In[1]::


batchSize = 16
learningRate = 1e-06
epoch = 5
print("Batch size = " + str(batchSize) + "\nLearning rate " + str(learningRate) + "\nEpochs "+str(epoch))


# In[2]:


#required libraries
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.layers import Dense
import numpy as np
import os


# In[3]:


#loading dataset
dataset = mnist.load_data("dataset.db")


# In[4]:


#splitting dataset
train, test = dataset


# In[5]:


#separating variables
xtrain, ytrain = train
xtest, ytest = test


# In[6]:


#flattening data variables
xtrain = xtrain.reshape(-1, 28*28).astype("float32")
xtest = xtest.reshape(-1,28*28).astype("float32")
ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)


# In[7]:


#model creation
model = Sequential()


# In[8]:


#layers
model.add(Dense(units=512, activation="relu", input_dim=28*28))


# In[9]:


model.add(Dense(units=256, activation="relu"))


# In[16]:


model.add(Dense(units=128, activation="relu"))


# In[11]:


model.add(Dense(units=32, activation="relu"))


# In[12]:


model.add(Dense(units=10, activation='softmax'))


# In[13]:


#compiling model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learningRate), metrics=['accuracy'])


# In[14]:


#training model
model.fit(xtrain, ytrain, epochs=epoch, batch_size=batchSize)


# In[15]:


acc = model.evaluate(xtest, ytest,)[1]*100


os.system("curl -u admin:yobah11111 http://192.168.172.3:8080/job/accGetter/build?token=getAccuracy")
print(acc)


# In[ ]:




