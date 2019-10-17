
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
import keras
from keras import layers
from keras import models
from keras import optimizers
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

# ### Class Declaration

# In[2]:


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
        print("Done")
        columns = ['class','order','family','genus','species_glc_id']
        self.df = pd.DataFrame(shuffle(self.df.values), columns=self.df.columns)
        try:
            self.embed_vectors1 = pkl.load(open("Data/Embed1.pkl","rb"))
        except:
            self.embed_vectors1 = {}
            for col_idx in range(len(columns)-1):
                x,y = [],[]
                print("Collecting " + columns[col_idx] + "," + columns[col_idx+1])
                x.extend([self.all_encoded[str(i)] for i in self.df[columns[col_idx]]])
                y.extend([self.all_encoded[str(i)] for i in self.df[columns[col_idx+1]]])
                x,y = np.array(x), np.array(y)
                print(x.shape, y.shape)
                print(np.max(x))
                model = Sequential()
                model.add(Embedding(input_dim=np.max(x)+1, output_dim=10, input_length=1, name="Embed"))
                model.add(Flatten())
                model.add(Dense(1, activation='relu'))
                model.compile(optimizer='nadam',loss='logcosh', metrics=['mae','accuracy'])
                model.summary()
                model.fit(x,y,epochs=30,batch_size=100)
                self.embed_vectors1[columns[col_idx]] = np.array(model.get_layer("Embed").get_weights()[0])
                del model
            pkl.dump(self.embed_vectors1, open("Data/Embed1.pkl","wb"))
        
        try:
            self.embed_vectors2 = pkl.load(open("Data/Embed2.pkl","rb"))
        except:
            self.embed_vectors2 = {}
            x,y = [self.all_encoded[str(i)] for i in self.df[columns[-1]]], [self.embed_vectors1[columns[-2]][self.all_encoded[str(i)]] for i in self.df[columns[-2]]]
            x,y = np.array(x), np.array(y)
            print(x.shape, y.shape)
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
            #if(cls not in ['Magnoliopsida']):
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
            c, o, f, g, s = self.all_encoded[l[3]], self.all_encoded[l[4]], self.all_encoded[l[5]], self.all_encoded[l[6]], self.all_encoded[l[7]] 
            self.y_train.append([c,o,f,g,s])
            
        for im in self.x_test:
            l = im.split("/")
            c, o, f, g, s = self.all_encoded[l[3]], self.all_encoded[l[4]], self.all_encoded[l[5]], self.all_encoded[l[6]], self.all_encoded[l[7]] #self.embed_vectors1['class'][self.all_encoded[l[3]]], self.embed_vectors1['order'][self.all_encoded[l[4]]], self.embed_vectors1['family'][self.all_encoded[l[5]]], self.embed_vectors1['genus'][self.all_encoded[l[6]]], self.embed_vectors2['species_glc_id'][int(l[7])]
            self.y_test.append([c,o,f,g,s])
        
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


# ### Implement data preprocessing

# In[3]:


data = Data_Preprocess()


# In[ ]:


data.ordered_call(root_dir="Data/Hierarchial Data/", csv_file="occurrences_train.csv")

p1 = pd.DataFrame(np.concatenate((data.y_train, data.y_test), axis=0))

p1 = p1.iloc[:10000,:]

p1.to_csv("values.tsv", sep="\t", index=False)

n1, n2 = [], []
for i in data.y_train:
    l = []
    for j in i:
        l.append(data.all_rev_encoded[j])
    n1.append(l)
n1 = np.array(n1)
for i in data.y_test:
    l = []
    for j in i:
        l.append(data.all_rev_encoded[j])
    n2.append(l)
n2 = np.array(n2)

p2 = pd.DataFrame(np.concatenate((np.array(["class","order","family","genus","species"]).reshape(1,5),n1,n2), axis=0))

p2 = p2.iloc[:10001,:]

p2.to_csv("words.tsv", sep="\t", index=False)p = pd.DataFrame(np.concatenate((data.embed_vectors1['class'], data.embed_vectors1['order'][-len(data.orders):], 
                                 data.embed_vectors1['family'][-len(data.family):], data.embed_vectors1['genus'][-len(data.genus):], 
                                 data.embed_vectors2['species_glc_id'][-len(data.species):]), axis=0))

p.to_csv("embeddings.tsv", sep="\t", index=False)

names = pd.DataFrame(np.array(list(data.all_encoded.keys())))

names.to_csv("names.tsv", sep="\t", index=False)
# In[ ]:


#data.y_train, data.y_test = np.array(data.y_train).reshape(-1,1), np.array(data.y_test).reshape(-1,1)
np.array(data.x_train).shape, np.array(data.y_train).shape, np.array(data.x_test).shape, np.array(data.y_test).shape


# # Model

# ### Class Declaration

# In[ ]:


class ImageDataGenerator(Sequence):
    
    def __init__(self, x_metadata, y_metadata, batch_size, crop_size, out_dim):
        self.x = x_metadata
        self.y = y_metadata
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.cp = crop_size
    
    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        x,y = [],[]
        for i in range(1,len(batch_y)):
            #x.append(np.transpose(tif.imread(batch_x[i])[:,self.cp:-self.cp,self.cp:-self.cp],(1,2,0)))
            x.append(np.transpose(tif.imread(batch_x[i])/255.0,(1,2,0)))
            l = []
            for j in range(len(batch_y[i])):
#                 t = np.zeros(self.out_dim)
#                 t[batch_y[i][j]] = 1
                l.append([batch_y[i][j]])
            y.append(np.array(l))
        return np.array(x), np.array(y)


# In[ ]:


class CNN_Model:
    
    def __init__(self, data_object, out_dim):
        self.img_height = 64
        self.img_width = 64
        self.img_channels = 33
        self.cardinality = 32
        self.data_object = data_object
        self.num_classes = out_dim

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
        class_pred = layers.Dense(128)(x)
        class_pred = layers.Dense(10, activation='softmax')(x)
        
        x1 = layers.RepeatVector(5)(class_pred)
        
        # LSTM for 5 timesteps to predict the embedded hierarchy
        x1 = layers.LSTM(50, return_sequences=True)(x1)
        x1 = layers.Dropout(0.2)(x1)
        x1 = layers.LSTM(50, return_sequences=True)(x1)
        x1 = layers.Dropout(0.2)(x1)
        x1 = layers.LSTM(50, return_sequences=True)(x1)
        x1 = layers.Dropout(0.2)(x1)
        x1 = layers.LSTM(50, return_sequences=True)(x1)
        x1 = layers.Dropout(0.2)(x1)
        x1 = layers.LSTM(50, return_sequences=True)(x1)
        x1 = layers.Dropout(0.2)(x1)
        x1 = layers.LSTM(50, return_sequences=True)(x1)
        x1 = layers.Dropout(0.2)(x1)
        x1 = layers.Dense(128, activation='relu')(x1)
        #x = layers.Dense(4096)(x)
        x1 = layers.Dense(self.num_classes, activation='softmax')(x1)

        return x1

    def model_create(self, time_steps, batch_size):
        image_tensor = layers.Input(shape=(self.img_height, self.img_width, self.img_channels))
        network_output = self.residual_network(image_tensor)
        model = models.Model(inputs=[image_tensor], outputs=[network_output])
        print(model.summary())
        # Compiling the CNN
        sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
        model.compile(optimizer = sgd, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy','mae'])
        return model
    
    def fit_generator(self, num_epochs=10, batch_size=32, crop_size=16, time_steps=5):        
        try:
            classifier = load_model("Code/Models/RCNN_ResNext2.h5")
        except:
            print("Training")
            classifier = self.model_create(time_steps=time_steps, batch_size=batch_size)
            train_data = ImageDataGenerator(self.data_object.x_train, self.data_object.y_train, batch_size, crop_size, self.num_classes)
            history = classifier.fit_generator(train_data, epochs=num_epochs, use_multiprocessing=True,shuffle=True)
            classifier.save("Code/Models/RCNN_ResNext2.h5")
        print("Testing")
        test_data = ImageDataGenerator(self.data_object.x_test, self.data_object.y_test, batch_size, crop_size, self.num_classes)
        scores = classifier.evaluate_generator(test_data, use_multiprocessing=True)
        print("Loss : ", scores[0])
        print("Metrics : ", scores[1:])
        return classifier


# ### Model Run

# In[ ]:


model_object = CNN_Model(data,len(data.all_encoded.keys()))


# In[ ]:


classifier = model_object.fit_generator(num_epochs=10, batch_size=16)


# # MRR metric
classifier = load_model("Code/Models/RCNN_ResNext.h5")
test_data = ImageDataGenerator(data.x_test[:10000], data.y_test[:10000], 32, 16)predictions = classifier.predict_generator(test_data)species_embedding = data.embed_vectors2['species_glc_id']y_test = np.array(data.y_test)[:,-1,:]preds = predictions[:,-1,:]
trues = [np.flatnonzero((species_embedding == i).all(1)) for i in y_test]from sklearn.metrics.pairwise import euclidean_distances

def prediction_to_species(ypred, unique_species_id, max_ranks=5):
    e2 = pkl.load(open("Data/Embed2.pkl", "rb"))
    species_embedding = e2['species_glc_id'][-3336:]

    embedding_distance = euclidean_distances(species_embedding, ypred.reshape(-1,10))
    top_indices = embedding_distance.flatten().argsort()[:max_ranks]
    species_glc_id = pd.Series(unique_species_id)[top_indices].reshape(-1,1)
    return species_glc_ids = 0
for i in range(len(trues)):
    if(i%100 == 1): print(i, s/i)
    try:
        ypred = prediction_to_species(preds[i],data.species, 100)
        r = ypred.index(trues[i])
    except:
        r = 3336
    s = s+(1.0/r)
print(s/i)