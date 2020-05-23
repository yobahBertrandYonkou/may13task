#!/usr/bin/env python
# coding: utf-8

# In[1].:


from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Flatten
from keras.layers import Dense
import pandas as pd
import random
import os


# In[11]:


random.seed()
batchSize = random.choice([16,32,64,128])
learningRate = random.choice([0.001,0.0001,0.00001,0.000001])
print("Batch size = " + str(batchSize) + "\nLearning rate " + str(learningRate))


# In[3]:


model = Sequential()


# In[4]:


model.add(Convolution2D(filters=64, kernel_size=(3,3), input_shape = (224,224,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(filters=32, kernel_size=(2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2,)))

model.add(Convolution2D(filters=16, kernel_size=(2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=2, activation='softmax'))


# In[ ]:





# In[5]:


train_data_dir = 'manwoman/train/'
validation_data_dir = 'manwoman/test/'

img_cols, img_rows = 224, 224

# Let's use some data augmentaiton 
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
# set our batch size (typically on most mid tier systems we'll use 16-32)
batch_size = batchSize
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')


# In[10]:


checkpoint = ModelCheckpoint("currentWeight.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

#earlystop = EarlyStopping(monitor = 'val_loss', 
#                         min_delta = 0, 
#                         patience = 3,
#                         verbose = 1,
#                         restore_best_weights = True)

# we put our call backs into a callback list
callbacks = [earlystop, checkpoint]

# We use a very small learning rate 
model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(lr = learningRate),
              metrics = ['accuracy'])

# Enter the number of training and validation samples here
nb_train_samples = 1613
nb_validation_samples = 346

# We only train 5 EPOCHS 
epochs = 5
batch_size = batchSize

history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)  


# In[9]:


log = pd.DataFrame(model.history.history)
log.to_csv('data.csv')
acc = round((log['val_accuracy'].iloc[-1]*100),2)
os.system("curl -u admin:yobah11111 http://192.168.172.3:8080/job/accGetter/build?token=getAccuracy")
print(acc)


# In[ ]:




