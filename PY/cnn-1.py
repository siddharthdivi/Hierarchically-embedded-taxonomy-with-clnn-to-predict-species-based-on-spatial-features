
# coding: utf-8

# In[34]:


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from multiprocessing import Pool
import pickle as pkl

# In[35]:


classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(66, (3, 3), input_shape = (64, 64, 33), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second layer convolutional
classifier.add(Convolution2D(33, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(3, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.summary()



'''# In[4]:

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Class wise Data/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32)

test_set = test_datagen.flow_from_directory('Class wise Data/test',
                                            target_size = (64, 64),
                                            batch_size = 32)

classifier.fit_generator(training_set,
                         samples_per_epoch = 50,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 20)
'''
# In[30]
with open("train_x.pkl","rb") as f:
    train_x = pkl.load(f)
    
with open("train_y.pkl","rb") as f:
    train_y = pkl.load(f)

with open("test_x.pkl","rb") as f:
    test_x = pkl.load(f)

with open("test_y.pkl","rb") as f:
    test_y = pkl.load(f)
# In[37]
classifier.fit(train_x,train_y,epochs=400,shuffle=True)

# In[]
classifier.evaluate(test_x,test_y)