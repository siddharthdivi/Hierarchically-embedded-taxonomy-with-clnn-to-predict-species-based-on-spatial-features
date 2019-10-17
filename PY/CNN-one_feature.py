
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
    
    train_xA = []
    train_yA = []
    train_xB = []
    train_yB = []
    test_x = []
    test_y = []

    def model_create(self):
        classifier = Sequential()

        # Step 1 - Convolution
        classifier.add(Convolution2D(96, (3, 3), input_shape = (32, 32, 33), activation = 'relu'))
        #classifier.add(MaxPooling2D(pool_size = (2, 2)))

        # Adding a second layer convolutional
        classifier.add(Convolution2D(288, (3, 3), activation = 'relu'))
        #classifier.add(MaxPooling2D(pool_size = (2, 2)))

        # Adding a third layer convolutional
        classifier.add(Convolution2D(864, (3, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        

        # Step 3 - Flattening
        classifier.add(Flatten())

        # Step 4 - Full connection
        classifier.add(Dense(128, activation = 'relu'))
        classifier.add(Dense(64, activation = 'relu'))
        classifier.add(Dense(1, activation = 'sigmoid'))

        # Compiling the CNN
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        classifier.summary()
        
        return classifier
        
    def reshaper(self,va, crop_size=16):
        t = []
        for i in range(va.shape[1]):
            d = []
            for j in range(va.shape[2]):
                l = []
                #for k in range(va.shape[0]):
                    l.append(va[23][i][j])
                d.append(l)
            t.append(d)
        f = []
        for i in range(crop_size,len(t)-crop_size):
            d = []
            for j in range(crop_size,len(t)-crop_size):
                d.append(t[i][j])
            f.append(d)
        f = np.array(f)/255
        return f
    
    def data_create(self):
        with open("Data/unique_classes.pkl","rb") as f:
            uc = pkl.load(f)  
        print(uc)
        for i in uc:
            if( i != "Magnoliopsida"):  
                print(i)
                for im in os.listdir("Data/Class wise Data/train/"+i):
                    self.train_yB.append(0)
                    self.train_xB.append((im,i))
        i = 0
        for im in os.listdir("Data/Class wise Data/train/Magnoliopsida"):
            if(i < len(self.train_xB)):                
                self.train_yA.append(1)
                self.train_xA.append((im,"Magnoliopsida"))
            else: break
            i+=1
        self.train_yB = np.array(self.train_yB)
        self.train_yA = np.array(self.train_yA)
        self.test_y = np.array(self.test_y)

        #self.test_y = self.test_y.reshape((self.test_y.shape[0],len(uc)))
        #self.train_y = self.train_y.reshape((self.train_y.shape[0],len(uc)))
        
    def model_run(self, batch_size=32, num_epochs=5, split=0.5):
        classifier = self.model_create()
        for e in range(num_epochs):
            acc = []
            for i in range(0,(len(self.train_xA)+len(self.train_xB)),batch_size):
                if(i+batch_size < (len(self.train_xA)+len(self.train_xB))):
                    print("Batch no : ",int(i/batch_size+1))
                    x = []
                    y = []
                    print("Class begins : ",self.train_xB[i][1],"\nClass ends : ",self.train_xA[i+batch_size][1])
                    for j in range(i,i+int(batch_size*split)):
                        x.append(self.reshaper(np.array(tif.imread("Data/Class wise Data/train/"+self.train_xB[j][1]+"/"+self.train_xB[j][0]))))
                    y+=list(self.train_yB[i:i+int(batch_size*split)])
                    for j in range(i+int(batch_size*split),i+batch_size):
                        x.append(self.reshaper(np.array(tif.imread("Data/Class wise Data/train/"+self.train_xA[j][1]+"/"+self.train_xA[j][0]))))
                    y+=list(self.train_yA[i+int(batch_size*split):i+batch_size])
                    x = np.array(x)
                    y = np.array(y)
                    print(x.shape, y.shape)
                    history = classifier.fit(x,y,epochs=1,batch_size=100,shuffle=True)
                    acc.extend(history.history['acc'])
            print(np.mean(acc))

# In[6]:


ob = CNN_model()


# In[7]:

ob.data_create()


# In[8]:


ob.model_run(batch_size=100, num_epochs=3)
