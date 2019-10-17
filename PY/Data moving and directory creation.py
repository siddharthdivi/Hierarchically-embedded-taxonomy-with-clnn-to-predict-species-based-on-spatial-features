
# coding: utf-8

# In[2]:


import os
import math
import shutil
import random
import numpy as np
import pandas as pd
import pickle as pkl
import tifffile as tif
from multiprocessing import Pool
from collections import OrderedDict


# In[3]:


i = 0
class data_preprocessing:
    
    uc = []
    splits = []
    im_class_map = {}
    train_class_map = {}
    test_class_map = {}
    
    def __init__(self):
        with open("../../Data/unique_classes.pkl","rb") as f:
            self.uc = pkl.load(f)
        self.uc = sorted(self.uc)
        self.im_class_map = {k: [] for k in self.uc}
        self.train_class_map = {k: [] for k in self.uc}
        self.test_class_map = {k: [] for k in self.uc}

        
    def create_directories_anew(self):
        try:
            shutil.rmtree("../../Data/Class wise Data")
            os.mkdir("../../Data/Class wise Data")
            os.mkdir("../../Data/Class wise Data/train")
            os.mkdir("../../Data/Class wise Data/test")
            for i in self.uc:
                os.mkdir("../../Data/Class wise Data/train/"+i)
                os.mkdir("../../Data/Class wise Data/test/"+i)
        except:
            os.mkdir("../../Data/Class wise Data")
            os.mkdir("../../Data/Class wise Data/train")
            os.mkdir("../../Data/Class wise Data/test")
            for i in self.uc:
                os.mkdir("../../Data/Class wise Data/train/"+i)
                os.mkdir("../../Data/Class wise Data/test/"+i)
    
    def create_directoy_class_map(self):
        df = pd.read_csv("../../occurrences_train.csv")
        for i in range(df.shape[0]):
            im_id = df['patch_id'][i]
            im_dir = df['patch_dirname'][i]
            cls = df['class'][i]
            self.im_class_map[cls].append((im_dir,im_id,cls))
        self.im_class_map = OrderedDict(sorted(self.im_class_map.items()))
    
    def splits_size_gen(self,train_size=0.85):
        for i in self.uc:
            self.splits.append(int(train_size*len(self.im_class_map[i])))
        
    def train_test_class_map(self):
        for k in self.im_class_map.keys():
            random.shuffle(self.im_class_map[k])
            for i in range(len(self.im_class_map[k])):
                if(i<self.splits[list(self.im_class_map.keys()).index(k)]):
                    self.train_class_map[k].append((self.im_class_map[k][i],0))
                else:
                    self.test_class_map[k].append((self.im_class_map[k][i],1))
        self.train_class_map = OrderedDict(sorted(self.train_class_map.items()))
        self.test_class_map = OrderedDict(sorted(self.test_class_map.items()))
    
    def paralleized_copying(self,item):
        global i
        i+=1
        if(i%100 == 0): print(i)
        p1 = item[0][0]
        p2 = item[0][1]
        p3 = item[0][2]
        if(item[1] == 0):
            p4 = "train"
        else:
            p4 = "test"
        source = "../../patchTrain/"+str(p1)+"/patch_"+str(p2)+".tif"
        dest = "../../Data/Class wise Data/"+str(p4)+"/"+str(p3)+"/"
        shutil.copy(source,dest)
        
    def copy_data(self):
        for k in self.train_class_map.keys():
            print(k," training copy")
            i=0
            with Pool(10) as p:
                p.map(self.paralleized_copying,self.train_class_map[k])
        for k in self.test_class_map.keys():
            print(k," testing copy")
            i=0
            with Pool(5) as p:
                p.map(self.paralleized_copying,self.test_class_map[k])


# In[4]:


ob = data_preprocessing()


# In[5]:


ob.create_directories_anew()


# In[6]:


ob.create_directoy_class_map()


# In[7]:


ob.splits_size_gen(train_size=0.7)


# In[8]:


ob.train_test_class_map()


# In[9]:


ob.copy_data()

