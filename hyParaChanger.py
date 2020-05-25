#!/usr/bin/env python
# coding: utf-8

# In[138]:


import random

#generates new values for hyperparameters
random.seed()
batchSize = random.choice([16,32,64,128])
learningRate = random.choice([0.001,0.0001,0.00001,0.000001])
epoch = random.choice([5,6,7,8,9,10])
print("Batch size = " + str(batchSize) + "\nLearning rate " + str(learningRate) + "\nEpoch "+ str(epoch))


# In[139]:


#gets the previous values
summary = open("summary.txt", "r")
data =str(summary.readlines()[0])
summary.close()
data = data.split(sep=" ")


# In[141]:


oldBVal = "batchSize = " + data[0]
newBVal = "batchSize = " + str(batchSize)
oldLrVal = "learningRate = "+ data[1]
newLrVal = "learningRate = "+ str(learningRate)
oldEpVal = "epoch = "+ data[2]
newEpVal = "epoch = "+ str(epoch)

#replaces hyperparamenters with a new value
def tuner(oldVal, newVal):
    reading_file = open("code.py", "r")
    new_file_content = ""
    for line in reading_file:
        stripped_line = line.strip()
        new_line = stripped_line.replace(oldVal, newVal,1)
        new_file_content += new_line +"\n"
    reading_file.close()

    writing_file = open("code.py", "w")
    writing_file.write(new_file_content)
    writing_file.close()


tuner(oldBVal, newBVal)
tuner(oldLrVal, newLrVal)
tuner(oldEpVal, newEpVal)

#stores present values into a file
values = str(batchSize) + " "+ str(learningRate) +" "+ str(epoch)

summary = open("summary.txt", "w")
summary.write(values)
summary.close()


# In[ ]:




