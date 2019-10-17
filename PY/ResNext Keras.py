
# coding: utf-8

# # Imports

# In[1]:


from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import math
import operator
import numpy as np
import pandas as pd
import pickle as pkl
import tifffile as tif
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.utils import shuffle
from keras import layers
from keras import models
from keras.utils import Sequence
from keras.models import Sequential
from keras.models import load_model, save_model
from keras.layers import Dense, Flatten, Embedding

# from keras.layers import Conv2D, Merge
# from keras.layers import Flatten, RepeatVector
# from keras.layers import MaxPool2D 
# from keras.layers import Reshape
# from collections import OrderedDict
# from keras.layers import TimeDistributed
# from keras.layers import LSTM
# from keras.layers import Permute, Embedding
# from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.layers.convolutional import Convolution2D, MaxPooling2D
# from keras.layers.normalization import BatchNormalization
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torchvision import transforms, utils
# from torch.utils.data import Dataset, DataLoader
os.chdir("../../")


# # Data Preprocessing

# ## Class Declaration

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
           
        return np.array([tif.imread(file_name)[:,self.cp:-self.cp,self.cp:-self.cp] 
                         for file_name in batch_x]), np.array(batch_y)


# In[3]:


class Data_Preprocess():
    
    def init_load(self, root_dir, csv_file):
        self.df = pd.read_csv(csv_file, low_memory=False)
        self.path = root_dir
    
    def create_mappings_for_unique_labels(self):
        # getting all unique names from csv file
        self.classes = list(sorted(self.df['class'].unique()))
        self.orders = list(sorted(self.df['order'].unique()))
        self.family = list(sorted(self.df['family'].unique()))
        self.genus = list(sorted(self.df['genus'].unique()))
        self.species = list(sorted(self.df['species_glc_id'].unique()))
        self.all_names = self.classes + self.orders + self.family + self.genus + self.species
        # creting map for one hot encoding / embedding
        self.all_encoded = {}
        self.all_rev_encoded = {}
        
        for i, name in enumerate(self.all_names):
            self.all_encoded[str(name)] = i
            self.all_rev_encoded[int(i)] = str(name)
        
    # embedding all the names
    def create_embedding(self):
        columns = ['class','order','family','genus','species_glc_id']
        self.df = pd.DataFrame(shuffle(self.df.values), columns=self.df.columns)
        try:
            self.embed_vectors1 = pkl.load(open("Data/Embed1.pkl","rb"))
        except:
            self.embed_vectors1 = {}
            for col_idx in range(len(columns)-1):
                print("Embedding " + columns[col_idx])
                x,y = [self.all_encoded[str(i)] for i in self.df[columns[col_idx]]], [self.all_encoded[str(i)] for i in self.df[columns[col_idx+1]]]
                x,y = np.array(x), np.array(y)
                print(x.shape, y.shape)
                print(max(x)+1)
                model = Sequential()
                model.add(Embedding(input_dim=max(x)+1, output_dim=10, input_length=1, name="Embed"))
                model.add(Flatten())
                model.add(Dense(1, activation='relu'))
                model.compile(optimizer='nadam',loss='logcosh', metrics=['mae','accuracy'])
                model.summary()
                model.fit(x,y,epochs=30,batch_size=250)
                self.embed_vectors1[columns[col_idx]] = np.array(model.get_layer("Embed").get_weights()[0])
                print(self.embed_vectors1[columns[col_idx]][:min(x)])
                del model
            pkl.dump(self.embed_vectors1,open("Data/Embed1.pkl","wb"))
        
        try:
            self.embed_vectors2 = pkl.load(open("Data/Embed2.pkl","rb"))
        except:
            self.embed_vectors2 = {}
            x,y = [self.all_encoded[str(i)] for i in self.df[columns[-1]]], [self.embed_vectors1[columns[-2]][self.all_encoded[str(i)]] for i in self.df[columns[-2]]]
            x,y = np.array(x), np.array(y)
            print(x.shape, y.shape)
            print(max(x)+1)
            model = Sequential()
            model.add(Embedding(input_dim=max(x)+1, output_dim=10, input_length=1, name="Embed"))
            model.add(Flatten())
            model.add(Dense(10))
            model.compile(optimizer='nadam',loss='logcosh', metrics=['mae','accuracy'])
            model.summary()
            model.fit(x,y,epochs=50,batch_size=200)
            self.embed_vectors2[columns[-1]] = np.array(model.get_layer("Embed").get_weights()[0])
            del model
            pkl.dump(self.embed_vectors2,open("Data/Embed2.pkl","wb"))
        
    def train_test_data_loading(self):
        self.x_train, self.x_test, self.y_train, self.y_test = [], [], [], []
        for cls in self.df['class'].unique():
            for order in self.df[self.df['class']==cls]['order'].unique():
                for family in self.df[(self.df['class']==cls) & (self.df['order']==order)]['family'].unique():
                    for genus in self.df[(self.df['class']==cls) & (self.df['order']==order) & (self.df['family']==family)]['genus'].unique():
                        for species in self.df[(self.df['class']==cls) & (self.df['order']==order) & (self.df['family']==family) & (self.df['genus']==genus)]['species_glc_id'].unique():
                            path = self.path+"train/"+cls+"/"+order+"/"+family+"/"+genus+"/"+str(species)+"/"
                            self.x_train.extend([path+i for i in os.listdir(path)])
                            path = self.path+"test/"+cls+"/"+order+"/"+family+"/"+genus+"/"+str(species)+"/"
                            self.x_test.extend([path+i for i in os.listdir(path)])
        
        np.random.shuffle(self.x_train)
        np.random.shuffle(self.x_test)
        
        for im in self.x_train:
            l = im.split("/")
            c, o, f, g, s = self.embed_vectors1['class'][self.all_encoded[l[3]]], self.embed_vectors1['order'][self.all_encoded[l[4]]], self.embed_vectors1['family'][self.all_encoded[l[5]]], self.embed_vectors1['genus'][self.all_encoded[l[6]]], self.embed_vectors2['species_glc_id'][int(l[7])]
            self.y_train.append(s)
            
        for im in self.x_test:
            l = im.split("/")
            c, o, f, g, s = self.embed_vectors1['class'][self.all_encoded[l[3]]], self.embed_vectors1['order'][self.all_encoded[l[4]]], self.embed_vectors1['family'][self.all_encoded[l[5]]], self.embed_vectors1['genus'][self.all_encoded[l[6]]], self.embed_vectors2['species_glc_id'][int(l[7])]
            self.y_test.append(s)
        
    def ordered_call(self, root_dir, csv_file):
        print("Creating the data preprocessing object and loading csv")
        self.init_load(root_dir, csv_file)
        print("Done!")
        print("Creating unique mappings for labels")
        self.create_mappings_for_unique_labels()
        print("Done!")
        print("Creating embeddings for all the names")
        self.create_embedding()
        print("Done!")
        print("Loading test and train image paths and corresponding labels")
        self.train_test_data_loading()
        print("Done!")


# ## Implement data preprocessing

# In[5]:


data = Data_Preprocess()


# In[6]:


data.ordered_call(root_dir="Data/Hierarchial Data/", csv_file="occurrences_train.csv")


# In[7]:


np.array(data.x_train).shape, np.array(data.y_train).shape, np.array(data.x_test).shape, np.array(data.y_test).shape


# # Model

# ## Class Declaration

# In[11]:


class CNN_Model:
    
    def __init__(self, data_object):
        self.img_height = 32
        self.img_width = 32
        self.img_channels = 33
        self.cardinality = 32
        self.data_object = data_object
        self.num_classes = 10

    def residual_network(self, x):
        """
        ResNeXt by default. For ResNet set `cardinality` = 1 above.

        """
        def add_common_layers(y):
            y = layers.BatchNormalization()(y)
            y = layers.LeakyReLU()(y)

            return y

        def grouped_convolution(y, nb_channels, _strides):
            # when `cardinality` == 1 this is just a standard convolution
            if self.cardinality == 1:
                return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)

            assert not nb_channels % self.cardinality
            _d = nb_channels // self.cardinality

            # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
            # and convolutions are separately performed within each group
            groups = []
            for j in range(self.cardinality):
                group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
                groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))

            # the grouped convolutional layer concatenates them as the outputs of the layer
            y = layers.concatenate(groups)

            return y

        def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
            """
            Our network consists of a stack of residual blocks. These blocks have the same topology,
            and are subject to two simple rules:
            - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
            - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
            """
            shortcut = y

            # we modify the residual building block as a bottleneck design to make the network more economical
            y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
            y = add_common_layers(y)

            # ResNeXt (identical to ResNet when `cardinality` == 1)
            y = grouped_convolution(y, nb_channels_in, _strides=_strides)
            y = add_common_layers(y)

            y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
            # batch normalization is employed after aggregating the transformations and before adding to the shortcut
            y = layers.BatchNormalization()(y)

            # identity shortcuts used directly when the input and output are of the same dimensions
            if _project_shortcut or _strides != (1, 1):
                # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
                # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
                shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)

            y = layers.add([shortcut, y])

            # relu is performed right after each batch normalization,
            # expect for the output of the block where relu is performed after the adding to the shortcut
            y = layers.LeakyReLU()(y)

            return y

        # conv1
        x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
        x = add_common_layers(x)

        # conv2
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        for i in range(3):
            project_shortcut = True if i == 0 else False
            x = residual_block(x, 128, 256, _project_shortcut=project_shortcut)

        # conv3
        for i in range(4):
            # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
            strides = (2, 2) if i == 0 else (1, 1)
            x = residual_block(x, 256, 512, _strides=strides)

        # conv4
        for i in range(6):
            strides = (2, 2) if i == 0 else (1, 1)
            x = residual_block(x, 512, 1024, _strides=strides)

        # conv5
        for i in range(3):
            strides = (2, 2) if i == 0 else (1, 1)
            x = residual_block(x, 1024, 2048, _strides=strides)

        x = layers.GlobalAveragePooling2D()(x)
        
        #x = layers.RepeatVector(5)(x)
        
        # LSTM for 5 timesteps to predict the embedded hierarchy
        #x = layers.LSTM(25)(x)
        x = layers.Dense(10)(x)

        return x

    def model_create(self, time_steps, batch_size):
        image_tensor = layers.Input(shape=(self.img_channels, self.img_height, self.img_width))
        network_output = self.residual_network(image_tensor)  
        model = models.Model(inputs=[image_tensor], outputs=[network_output])
        print(model.summary())
        # Compiling the CNN
        model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['accuracy','mae'])
        return model
    
    def fit_generator(self, num_epochs=10, batch_size=32, crop_size=16, time_steps=5):        
        try:
            classifier = load_model("Code/Models/RCNN_ResNext.h5")
        except:
            print("Training")
            classifier = self.model_create(time_steps=time_steps, batch_size=batch_size)
            train_data = ImageDataGenerator(self.data_object.x_train, self.data_object.y_train, batch_size, crop_size)
            history = classifier.fit_generator(train_data, epochs=num_epochs, use_multiprocessing=True,shuffle=True)
            classifier.save("Code/Models/RCNN_ResNext.h5")
        print("Testing")
        test_data = ImageDataGenerator(self.data_object.x_test, self.data_object.y_test, batch_size, crop_size)
        scores = classifier.evaluate_generator(test_data, use_multiprocessing=True)
        print("Loss : ", scores[0])
        print("Metrics : ", scores[1:])
        return classifier


# ## Model Run

# In[12]:


model_object = CNN_Model(data)


# In[ ]:


classifier = model_object.fit_generator(num_epochs=10, batch_size=30)

