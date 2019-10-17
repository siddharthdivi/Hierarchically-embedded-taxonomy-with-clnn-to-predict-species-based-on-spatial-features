
# coding: utf-8

# In[1]:


import os
import math
import random
import numpy as np
import pickle as pkl
import tifffile as tif
from PIL import Image
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[5]:


class CNN_model:
    
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    def model_create(self):
        classifier = Sequential()

        # Step 1 - Convolution
        classifier.add(Convolution2D(96, (3, 3), input_shape = (64, 64, 33), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))

        # Adding a second layer convolutional
        classifier.add(Convolution2D(288, (3, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))

        # Adding a third layer convolutional
        classifier.add(Convolution2D(864, (3, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        
        # Adding a fourth layer convolutional
        classifier.add(Convolution2D(2592, (3, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        
        # Step 3 - Flattening
        classifier.add(Flatten())

        # Step 4 - Full connection
        classifier.add(Dense(256, activation = 'relu'))
        classifier.add(Dense(128, activation = 'relu'))
        classifier.add(Dense(64, activation = 'relu'))
        classifier.add(Dense(32, activation = 'relu'))
        classifier.add(Dense(10, activation = 'sigmoid'))

        # Compiling the CNN
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        classifier.summary()
        
        return classifier
        
    def reshaper(self,va):
        t = []
        for i in range(va.shape[1]):
            d = []
            for j in range(va.shape[2]):
                l = []
                for k in range(va.shape[0]):
                    l.append(va[k][i][j])
                d.append(l)
            t.append(d)
        return np.array(t)/255.0
    
    def data_create(self):
        with open("Data/unique_classes.pkl","rb") as f:
            uc = pkl.load(f)
        for i in uc:
            for im in os.listdir("Data/Class wise Data/train/"+i):
                self.train_x.append((im,i))
                l = np.zeros((1,len(uc)))
                l[0][uc.index(i)] = 1
                self.train_y.append(l)
            for im in os.listdir("Data/Class wise Data/train/"+i):
                self.test_x.append((im,i))
                l = np.zeros((1,len(uc)))
                l[0][uc.index(i)] = 1
                self.test_y.append(l)

        self.train_y = np.array(self.train_y)
        self.test_y = np.array(self.test_y)

        self.test_y = self.test_y.reshape((self.test_y.shape[0],len(uc)))
        self.train_y = self.train_y.reshape((self.train_y.shape[0],len(uc)))
        
    def model_run(self, batch_size=32, num_epochs=100):
        classifier = self.model_create()
        for i in range(0,len(self.train_x),batch_size):
            print("Batch no : ",int(i/batch_size+1))
            x = []
            if(i+batch_size >= len(self.train_x)):
                print("Class begins : ",self.train_x[i][1],"\nClass ends : ",self.train_x[len(self.train_x)-1][1])
                for j in range(i,len(self.train_x)):
                    x.append(self.reshaper(np.array(tif.imread("Data/Class wise Data/train/"+self.train_x[j][1]+"/"+self.train_x[j][0]))))
                y = self.train_y[i:len(self.train_x)]
            else:
                print("Class begins : ",self.train_x[i][1],"\nClass ends : ",self.train_x[i+batch_size][1])
                for j in range(i,i+batch_size):
                    x.append(self.reshaper(np.array(tif.imread("Data/Class wise Data/train/"+self.train_x[j][1]+"/"+self.train_x[j][0]))))
                y = self.train_y[i:i+batch_size]
            x = np.array(x)
            #print(x.shape, y.shape)
            classifier.fit(x,y,epochs=num_epochs,shuffle=True)


# In[6]:


ob = CNN_model()


# In[7]:

ob.data_create()


# In[8]:


ob.model_run(batch_size=150, num_epochs=3)
