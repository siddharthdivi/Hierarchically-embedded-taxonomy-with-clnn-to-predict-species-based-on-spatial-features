
# coding: utf-8

# In[ ]:


import os
import sys
import math
import operator
import numpy as np
import pandas as pd
import pickle as pkl
import tifffile as tif
from keras.layers import Dense
from keras.layers import Conv2D, Merge
from multiprocessing import Pool
from keras.utils import Sequence
from keras.layers import Flatten, RepeatVector
from keras.layers import MaxPool2D
from keras.models import Sequential 
from keras.layers import Reshape
from collections import OrderedDict
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import Permute
from keras.models import load_model, save_model
os.chdir("../../")


# In[ ]:


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
           
        return [np.array([np.transpose(tif.imread(file_name)/255.0,(1,2,0))
                         [self.cp:-self.cp,self.cp:-self.cp,:] for file_name in batch_x]),
               np.array([np.transpose(tif.imread(file_name)/255.0,(1,2,0))
                         [self.cp:-self.cp,self.cp:-self.cp,:] for file_name in batch_x]),
                np.array([np.transpose(tif.imread(file_name)/255.0,(1,2,0))
                         [self.cp:-self.cp,self.cp:-self.cp,:] for file_name in batch_x])], np.array(batch_y)         

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
            self.train_seq_y.append([[c],[o],[f],[g],[s]])
            
        for p in self.test_pathdata_x:
            y = p.split("/")
            c = int(y[3])
            o = int(y[4])
            f = int(y[5])
            g = int(y[6])
            s = int(y[7])
            self.train_seq_y.append([[c],[o],[f],[g],[s]])
    
    def onehot_output(self):
        for sp in self.species:
            y = np.zeros(len(self.species))
            y[list(self.species).index(sp)] = 1
            self.onehot[sp] = y
            
    def model_create(self, time_steps=5, batch_size=32):
        
        cls1 = Sequential()
        cls2 = Sequential()
        cls3 = Sequential()
        # Branch 1
        cls1.add(Conv2D(filters=66, kernel_size=(1, 1), input_shape=(32,32,33), activation = 'relu'))
        cls1.add(Conv2D(filters=96, kernel_size=(2, 2), activation = 'relu'))
        cls1.add(Conv2D(filters=128, kernel_size=(3,3), activation = 'relu'))
        cls1.add(Flatten())
        # Branch 2
        cls2.add(Conv2D(filters=66, kernel_size=(1,1), input_shape=(32,32,33), activation = 'relu'))
        cls2.add(Conv2D(filters=96, kernel_size=(2,2), activation = 'relu'))
        cls2.add(Conv2D(filters=128, kernel_size=(3,3), activation = 'relu'))
        cls2.add(Conv2D(filters=64, kernel_size=(3,3), activation = 'relu'))
        cls2.add(Conv2D(filters=128, kernel_size=(4,4), activation = 'relu'))
        cls2.add(Flatten())
        # Branch 3
        cls3.add(Conv2D(filters=66, kernel_size=(1,1), input_shape=(32,32,33), activation = 'relu'))
        cls3.add(Conv2D(filters=96, kernel_size=(2,2), activation = 'relu'))
        cls3.add(Conv2D(filters=128, kernel_size=(2,2), activation = 'relu'))
        cls3.add(Conv2D(filters=256, kernel_size=(3,3), activation = 'relu'))
        cls3.add(Conv2D(filters=128, kernel_size=(3,3), activation = 'relu'))
        cls3.add(Conv2D(filters=256, kernel_size=(5,5), activation = 'relu'))
        cls3.add(Conv2D(filters=128, kernel_size=(5,5), activation = 'relu'))
        cls3.add(Flatten())
        
        classifier = Sequential()
        classifier.add(Merge([cls1,cls2,cls3], mode='concat'))
        classifier.add(RepeatVector(time_steps))
        classifier.add(LSTM(100, return_sequences=True))
        # Step 4 - Full connection
        classifier.add(Dense(256, activation = 'relu'))
        classifier.add(Dense(128, activation = 'relu'))
        classifier.add(Dense(1, activation = 'relu'))
        # Compiling the CNN
        classifier.compile(optimizer = 'Nadam', loss = 'logcosh', metrics = ['mae'])
        classifier.summary()
        return classifier
    
    def fit_generator(self, num_epochs=10, batch_size=32, crop_size=16, time_steps=5):        
        try:
            classifier = load_model("Code/Models/CNN-RNN_1.h5")
        except:
            print("Training")
            classifier = self.model_create(time_steps=time_steps, batch_size=batch_size)
            train_data = ImageDataGenerator(self.train_pathdata_x, self.train_seq_y, batch_size, crop_size)
            history = classifier.fit_generator(train_data, epochs=num_epochs, use_multiprocessing=True,shuffle=True)
            classifier.save("Code/Models/CNN-RNN_1.h5")
        print("Testing")
        test_data = ImageDataGenerator(self.test_pathdata_x, self.test_seq_y, batch_size, crop_size)
        scores = classifier.evaluate_generator(test_data, use_multiprocessing=True)
        print("Loss : ", scores[0])
        print("Accuracy : ", scores[1])


# In[ ]:


ob = CNN_Model("Data/Hierarchial Data/")


# In[ ]:


ob.fit_generator(num_epochs=10, batch_size=30, time_steps=5)

