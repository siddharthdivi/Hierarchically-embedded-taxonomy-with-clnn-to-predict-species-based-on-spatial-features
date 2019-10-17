
# coding: utf-8

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
'''
class ValidationImageGenerator(Sequence):
    
    def __init__(self, x_metadata,batch_size, crop_size):
        self.x = x_metadata
        self.batch_size = batch_size
        self.cp = crop_size
    
    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
           
        return np.array([tif.imread(file_name)[:,self.cp:-self.cp,self.cp:-self.cp] 
                         for file_name in batch_x])

class Val_load():
    
    def init_load(self, root_dir):
        self.path = root_dir
    
    def validation_data_loading(self,root_dir):
        self.x_val=[]
        self.y_val=[]
        for filename in os.listdir(root_dir):
            for image in os.listdir(root_dir+'/'+str(filename)):
                self.x_val.extend([root_dir+'/'+str(filename)+'/'+str(image)])
                self.y_val.append(image)

val_data = Val_load()
val_data.validation_data_loading(root_dir='patchTest')
val_data.x_val=np.array(val_data.x_val)

#Total test images
val_data.x_val.shape

validation_x=ValidationImageGenerator(val_data.x_val,32,16)

classifier = load_model("Code/Models/RCNN_ResNext.h5")

predictions=classifier.predict_generator(validation_x)

#if it returns nx5x10
preds=predictions[:,-1,:]

preds.shape

p1 = pd.DataFrame([int(i.split(".")[0].split("_")[1]) for i in val_data.y_val], columns=['patch_id'])
p2 = pd.DataFrame(preds, columns=[i+1 for i in range(predictions.shape[2])])

p = pd.concat([p1,p2], axis=1)

p.to_csv("Data/Test_Predictions.csv", sep=",")
'''


# In[2]:


p = pd.read_csv("Data/Test_Predictions.csv")


# In[22]:


import pickle as pkl
from sklearn.metrics.pairwise import euclidean_distances

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def prediction_to_species(p, max_ranks=5):
    """
    Function that will take in the prediction of the CNN-LSTM model and get the species id.

    Inputs :
    ypred : Vector of dimensions n x 10.

    Outputs :
    DataFrame of type :
    patch_id (int), species_glc_id(int) , probability(float) , rank(int) .

    """

    e2 = pkl.load(open("Data/Embed2.pkl", "rb"))
    #print ("Size of the species embedding pickle : ", e2.shape)
    ypred = p.values[:,2:]
    # Species embeddings from indices 0 to 3335, of shape 3336.
    species_embedding = e2['species_glc_id'][-3336:, :]
    df = pd.read_csv("occurrences_train.csv", low_memory=False)
    df = df[["class", "order", "family", "genus", "species_glc_id"]]
    unique_species_id = df['species_glc_id'].unique()
    masterdf = pd.DataFrame()

    cntr = 0
    for species in ypred:
        
        if(cntr % 1000 == 0):
            print(cntr)
        # Now compare this vector with each vector in species_embedding.
        '''
        GET THE PATCH FROM THE NAME OF THE IMAGES.
        patch_id is a list of patch names for the test images.
        '''
        patch_id = np.array([p.values[cntr,1]]*max_ranks).reshape(-1,1)
        cntr+=1
        embedding_distances = euclidean_distances(species_embedding, species.reshape(-1,10))
        top_indices = embedding_distances.flatten().argsort()[:max_ranks]
        top_distances = embedding_distances[top_indices].tolist()
        top_distances = softmax(np.array([1.0/i[0] for i in top_distances])).reshape(-1,1)
        rank = np.array(range(1,len(top_indices)+1)).reshape(-1,1)
        # Now get the species_glc_id , from the numpy array of unique species.

        # Use the top_indices, to get the species_glc_id from unique_species_id.
        species_glc_id = unique_species_id[top_indices].reshape(-1,1)
        masterdf = pd.concat([masterdf, pd.DataFrame(
            np.concatenate((patch_id, species_glc_id, top_distances, rank), axis=1))], axis=0)

    #Writing the dataframe to a csv.
    masterdf.columns = ["patch_id","species_glc_id","probability","rank"]
    masterdf["patch_id"] = masterdf["patch_id"].astype(int)
    masterdf["species_glc_id"] = masterdf["species_glc_id"].astype(int)
    masterdf["probability"] = masterdf["probability"].astype(float)
    masterdf["rank"] = masterdf["rank"].astype(int)
    return masterdf


# In[27]:


num_ranks = int(input("How many ranks for each image : "))
breakpt = 20000

# In[28]:


mpd = pd.DataFrame()
for i in range(0,p.shape[0],breakpt):
    print("Main Iter : ", i/breakpt)
    if(i==0):
        mpd = prediction_to_species(p.iloc[i:i+breakpt], num_ranks)
    else:
        mpd = pd.concat([mpd, prediction_to_species(p[i:i+breakpt], num_ranks)], axis=0)
mpd.to_csv("MLRG_SSN3_run100.csv",sep=";",index=False,header=False)