
# coding: utf-8

# In[1]:


import os
import sys
import math
import operator
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
from keras.layers import Reshape
from collections import OrderedDict
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.models import load_model, save_model
os.chdir("../../")


# In[2]:


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
        
        self.onehot = {}
        self.path = directory
        
        df = pd.read_csv("occurrences_train.csv",low_memory=False)
        with open("Data/hierarchy_data.pkl","rb") as f:
            hd = pkl.load(f)
        with open("Data/class_encoding.pkl","rb") as f:
            self.classes = pkl.load(f)
        with open("Data/order_encoding.pkl","rb") as f:
            self.orders = pkl.load(f)
        with open("Data/family_encoding.pkl","rb") as f:
            self.families = pkl.load(f)
        with open("Data/genus_encoding.pkl","rb") as f:
            self.genuses = pkl.load(f)
        with open("Data/specie_encoding.pkl","rb") as f:
            self.species = pkl.load(f)

        self.onehot_output()

        self.train_pathdata_x = []
        self.train_seq_y = []
        self.test_pathdata_x = []
        self.test_seq_y = []
        
        for cls in hd.keys():
            for order in hd[cls].keys():
                for family in hd[cls][order].keys():
                    for genus in hd[cls][order][family].keys():
                        for specie in hd[cls][order][family][genus]:
                            for im in os.listdir(self.path+"train/"+str(self.classes[cls])+"/"+str(self.orders[order])
                                                 +"/"+str(self.families[family])+"/"+str(self.genuses[genus])+"/"+str(specie)):
                                self.train_pathdata_x.append(self.path+"train/"+str(self.classes[cls])+"/"+str(self.orders[order])
                                                             +"/"+str(self.families[family])+"/"+str(self.genuses[genus])+"/"+str(specie)+"/"
                                                             +im)
                            
        for cls in hd.keys():
            for order in hd[cls].keys():
                for family in hd[cls][order].keys():
                    for genus in hd[cls][order][family].keys():
                        for specie in hd[cls][order][family][genus]:
                            for im in os.listdir(self.path+"test/"+str(self.classes[cls])+"/"+str(self.orders[order])
                                                 +"/"+str(self.families[family])+"/"+str(self.genuses[genus])+"/"+str(specie)):
                                self.test_pathdata_x.append(self.path+"test/"+str(self.classes[cls])+"/"+str(self.orders[order])
                                                             +"/"+str(self.families[family])+"/"+str(self.genuses[genus])+"/"+str(specie)+"/"
                                                             +im)
        
        np.random.shuffle(self.train_pathdata_x)
        np.random.shuffle(self.test_pathdata_x)
        
        for p in self.train_pathdata_x:
            y = p.split("/")
            c = int(y[3])
            o = int(y[4])
            f = int(y[5])
            g = int(y[6])
            s = int(y[7])
            self.train_seq_y.append(self.onehot[s])
            
        for p in self.test_pathdata_x:
            y = p.split("/")
            c = int(y[3])
            o = int(y[4])
            f = int(y[5])
            g = int(y[6])
            s = int(y[7])
            self.train_seq_y.append(self.onehot[s])
    
    def onehot_output(self):
        for sp in self.species:
            y = np.zeros(len(self.species))
            y[list(self.species).index(sp)] = 1
            self.onehot[sp] = y
            
    def model_create(self, time_steps=1, batch_size=32):
        
        classifier = Sequential()
        # Step 1 - Convolution
        classifier.add(Conv2D(filters=96, kernel_size=(2, 2), input_shape = (32,32,33), activation = 'relu'))
        classifier.add(Conv2D(filters=288, kernel_size=(2, 2), activation = 'relu'))
        classifier.add(Conv2D(filters=864, kernel_size=(2, 2), activation = 'relu'))
        #classifier.add(Conv2D(filters=, kernel_size=(2, 2), activation = 'relu'))
        classifier.add(MaxPool2D(pool_size = (2, 2)))
        # Step 3 - Flattening
        classifier.add(Flatten())
        # Step 4 - Full connection
        #classifier.add(Dense(6500, activation = 'relu'))
        #classifier.add(Dense(6000, activation = 'relu'))
        #classifier.add(Dense(5500, activation = 'relu'))
        #classifier.add(Dense(5000, activation = 'relu'))
        #classifier.add(Dense(4500, activation = 'relu'))
        #classifier.add(Dense(, activation = 'relu'))
        classifier.add(Dense(128, activation = 'relu'))
        classifier.add(Dense(3336, activation = 'softmax'))
        # Compiling the CNN
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        classifier.summary()
        return classifier
    
    def fit_generator(self, num_epochs=10, batch_size=32, crop_size=16, time_steps=5):        
        try:
            classifier = load_model("Code/Models/CNN-RNN_1.h5")
        except:
            classifier = self.model_create(time_steps=time_steps, batch_size=batch_size)
            train_data = ImageDataGenerator(self.train_pathdata_x, self.train_seq_y, batch_size, crop_size)
            history = classifier.fit_generator(train_data, epochs=num_epochs, use_multiprocessing=True,shuffle=True)
            classifier.save("Code/Models/CNN-RNN_1.h5")
        train_data = ImageDataGenerator(self.test_pathdata_x, self.test_seq_y, batch_size, crop_size)
        scores = classifier.evaluate_generator(test_data, use_multiprocessing=True)
        print("Loss : ", scores[0])
        print("Accuracy : ", scores[1])


# In[3]:


ob = CNN_Model("Data/Hierarchial Data/")


# In[ ]:


ob.fit_generator(num_epochs=10, batch_size=100, time_steps=1)

