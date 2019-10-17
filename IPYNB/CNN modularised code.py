
# coding: utf-8

# In[1]:


import os
import sys
import math
import numpy as np
import pandas as pd
import pickle as pkl
import tifffile as tif
from keras.layers import Dense
from keras.layers import Conv2D
from multiprocessing import Pool
from keras.utils import Sequence
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.models import Sequential
os.chdir("../../")


# In[11]:


class ImageDataGenerator(Sequence):
    
    def __init__(self, x_metadata, y_metadata, batch_size, crop_size):
        self.x = x_metadata
        self.y = y_metadata
        self.batch_size = batch_size
        self.cp = crop_size
    
    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        return np.array([np.transpose(np.array(tif.imread(file_name), dtype=int)/255.0,
                                     (1,2,0))[self.cp:-self.cp,self.cp:-self.cp,:]
                        for file_name in batch_x]), np.array(batch_y)         

class CNN_Model:
    
    def __init__(self, directory):
        
        self.path = directory
        self.classes = sorted(list(os.listdir(self.path+"/train")))
        self.train_metadata_x = []
        self.train_metadata_y = []
        self.test_metadata_x = []
        self.test_metadata_y = []
        
        for cls in self.classes:
            for im in os.listdir(self.path+"/train/"+cls):
                self.train_metadata_x.append(self.path+"/train/"+cls+"/"+im)
        
        for cls in self.classes:
            for im in os.listdir(self.path+"/test/"+cls):
                self.test_metadata_x.append(self.path+"/test/"+cls+"/"+im)
        
        np.random.shuffle(self.train_metadata_x)
        np.random.shuffle(self.test_metadata_x)
        
        for p in self.train_metadata_x:
            y = np.zeros(10, dtype=int)
            y[self.classes.index(p.split("/")[3])] = 1
            self.train_metadata_y.append(y)
            
        for p in self.test_metadata_x:
            y = np.zeros(10, dtype=int)
            y[self.classes.index(p.split("/")[3])] = 1
            self.test_metadata_y.append(y)
    
    def model_create(self):
        
        classifier = Sequential()
        # Step 1 - Convolution
        classifier.add(Conv2D(filters=96, kernel_size=(2, 2), input_shape = (32,32,33), activation = 'relu'))
        classifier.add(Conv2D(filters=288, kernel_size=(2, 2), activation = 'relu'))
        classifier.add(Conv2D(filters=576, kernel_size=(2, 2), activation = 'relu'))
        classifier.add(Conv2D(filters=64, kernel_size=(2, 2), activation = 'relu'))
        classifier.add(MaxPool2D(pool_size = (2, 2)))
        # Step 3 - Flattening
        classifier.add(Flatten())
        # Step 4 - Full connection
        classifier.add(Dense(128, activation = 'relu'))
        classifier.add(Dense(128, activation = 'tanh'))
        classifier.add(Dense(10, activation = 'sigmoid'))
        # Compiling the CNN
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        classifier.summary()
        return classifier
    
    def fit_generator(self, num_epochs=10, batch_size=32, crop_size=16):        
        classifier = self.model_create()
        train_data = ImageDataGenerator(self.train_metadata_x, self.train_metadata_y, batch_size, crop_size)
        test_data = ImageDataGenerator(self.test_metadata_x, self.test_metadata_y, batch_size, crop_size)
        history = classifier.fit_generator(train_data, epochs=num_epochs, use_multiprocessing=True,shuffle=True)
        classifier.evaluate_generator(test_data, use_multiprocessing=True)


# In[12]:


ob = CNN_Model(directory="Data/Class wise Data")


# In[ ]:


ob.fit_generator(num_epochs=15, batch_size=100, crop_size=16)

