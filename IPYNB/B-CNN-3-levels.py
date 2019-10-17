
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
from keras import optimizers
from keras import backend as K
from multiprocessing import Pool
from keras.utils import Sequence
from collections import OrderedDict
from keras.initializers import he_normal
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.models import Sequential, Model, load_model, save_model 
from keras.layers import Dropout, Activation, Dense, Conv2D, Merge, Flatten, RepeatVector, MaxPool2D, Input
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
        
        y1, y2, y3 = [], [], []
        
        for t in batch_y:
            y1.append(t[0])
            y2.append(t[1])
            y3.append(t[2])
           
        return [np.array([np.transpose(tif.imread(file_name)/255.0,(1,2,0))[self.cp:-self.cp,self.cp:-self.cp,:] 
                          for file_name in batch_x])], [np.array(y1),np.array(y2),np.array(y3)]         

class CNN_Model:
    
    def __init__(self, directory):
        
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

        self.train_pathdata_x = []
        self.train_seq_y = []
        self.test_pathdata_x = []
        self.test_seq_y = []
        
        print("Getting paths to images")
        for cls in hd.keys():
            print(cls)
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
        
        print("Done getting paths")
        np.random.shuffle(self.train_pathdata_x)
        np.random.shuffle(self.test_pathdata_x)
        
        print("Getting labels for images")
        for p in self.train_pathdata_x:
            y = p.split("/")
            c = int(y[3])
            o = int(y[4])
            f = int(y[5])
            g = int(y[6])
            s = int(y[7])
            z1 = np.zeros(len(self.classes.keys()))
            z1[c] = 1.0
            z2 = np.zeros(len(self.orders.keys()))
            z2[o] = 1.0
            z3 = np.zeros(len(self.species))
            z3[s-1] = 1.0
            self.train_seq_y.append([z1,z2,z3])
            
        for p in self.test_pathdata_x:
            y = p.split("/")
            c = int(y[3])
            o = int(y[4])
            f = int(y[5])
            g = int(y[6])
            s = int(y[7])
            z1 = np.zeros(len(self.classes.keys()))
            z1[c] = 1.0
            z2 = np.zeros(len(self.orders.keys()))
            z2[o] = 1.0
            z3 = np.zeros(len(self.species))
            z3[s-1] = 1.0
            self.test_seq_y.append([z1,z2,z3])
        print("Done getting labels")
            
    def model_create(self, batch_size=32):
        
        alpha = K.variable(value=0.98, dtype="float32", name="alpha") # A1 in paper
        beta = K.variable(value=0.01, dtype="float32", name="beta") # A2 in paper
        gamma = K.variable(value=0.01, dtype="float32", name="gamma") # A3 in paper

        img_input = Input(shape=(32,32,33), name='input')
        #--- block 1 ---
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        #--- block 2 ---
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        #--- coarse 1 branch ---
        c_1_bch = Flatten(name='c1_flatten')(x)
        c_1_bch = Dense(256, activation='relu', name='c1_fc_cifar10_1')(c_1_bch)
        c_1_bch = BatchNormalization()(c_1_bch)
        c_1_bch = Dropout(0.5)(c_1_bch)
        c_1_bch = Dense(256, activation='relu', name='c1_fc2')(c_1_bch)
        c_1_bch = BatchNormalization()(c_1_bch)
        c_1_bch = Dropout(0.5)(c_1_bch)
        c_1_pred = Dense(len(self.classes.keys()), activation='softmax', name='class')(c_1_bch)

        #--- block 3 ---
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        #--- coarse 2 branch ---
        c_2_bch = Flatten(name='c2_flatten')(x)
        c_2_bch = Dense(1024, activation='relu', name='c2_fc_cifar100_1')(c_2_bch)
        c_2_bch = BatchNormalization()(c_2_bch)
        c_2_bch = Dropout(0.5)(c_2_bch)
        c_2_bch = Dense(1024, activation='relu', name='c2_fc2')(c_2_bch)
        c_2_bch = BatchNormalization()(c_2_bch)
        c_2_bch = Dropout(0.5)(c_2_bch)
        c_2_pred = Dense(len(self.orders.keys()), activation='softmax', name='order')(c_2_bch)

        #--- block 4 ---
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool')(x)


        #--- block 5 ---
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = BatchNormalization()(x)

        #--- fine block ---
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc_cifar100_1')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu', name='fc_cifar100_2')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        fine_pred = Dense(len(self.species), activation='softmax', name='species')(x)

        model = Model(img_input, [c_1_pred, c_2_pred, fine_pred], name='IMAGECLEF18 hierarchy')
        
        #----------------------- compile and fit ---------------------------
        sgd = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', 
                      optimizer=sgd, 
                      loss_weights=[alpha, beta, gamma], 
                      # optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        model.summary()
        return model
    
    def fit_generator(self, num_epochs=10, batch_size=32, crop_size=16):        
        try:
            classifier = load_model("Code/Models/CNN-RNN_1.h5")
        except:
            classifier = self.model_create(batch_size=batch_size)
            train_data = ImageDataGenerator(self.train_pathdata_x, self.train_seq_y, batch_size, crop_size)
            print("Training")
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


ob.fit_generator(num_epochs=10, batch_size=30)

