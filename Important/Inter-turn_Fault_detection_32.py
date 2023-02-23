# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gc
import memory_profiler
import numpy
import cupy as np
import matplotlib.pyplot as plt
import PIL
import scipy
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import sys
import cv2
import glob
import os
import time
from tensorflow.keras.layers import Conv3D,Activation,Conv2D, ConvLSTM2D, MaxPooling2D, MaxPooling3D, BatchNormalization, Flatten, Input, Dense, GRU, Embedding, LSTM, SimpleRNN, Dropout, Bidirectional, TimeDistributed
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.optimizers import RMSprop,Adam,SGD,Adadelta,Adagrad,Adamax
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.backend.tensorflow_backend import set_session
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow import keras
from matplotlib import style
#style.use('classic')
from joblib import Parallel, delayed
import pywt
from numpy import genfromtxt
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
from PIL import Image 
from scipy.signal import resample
from io import StringIO
from CWT_GPU.wavelets import Morlet
from CWT_GPU.transform import WaveletTransformTorch
from CWT_GPU import cws
from CWT_GPU.wfun import periods2scales


# %%
from tensorflow.python.platform import build_info as tf_build_info
print("Tensorflow verison: ",tf.__version__)
print("CUDA verison: ", tf_build_info.cuda_version_number)
print("CUDNN verison: ", tf_build_info.cudnn_version_number)


# %%
from tensorflow.python.client import device_lib

config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
print(device_lib.list_local_devices())

tf.config.experimental.list_physical_devices('GPU')

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# %% [markdown]
# ## Build Dataset

# %%
percentageShort = np.arange(0.0001,1,1/400)*100
percentageShort = np.around(percentageShort, decimals=2, out=None)
pShort = numpy.concatenate((percentageShort, percentageShort, percentageShort, percentageShort, percentageShort, percentageShort), axis=0)

default_path=r"C:\Users\hp\iCloudDrive\Final Year Project\Matlab\Windings"
phase = ['phaseA', 'phaseB', 'phaseC']
phaseBool = ['100','010','001']
winding = ['Primary','Secondary']
windingBool = ['10','01']
faultBool = ['0','1']
section = ['Va', 'Vb', 'Vc']
parameter = ['Voltage','Current']
name = ['output_voltage','output_current']
ext = '.csv'
im_ext = '.png'
dataset1=[]
dataset2=[]
dataset3=[]
dataset4=[]
dataset5=[]
dataset6=[]

locationDataset = []
magnitudeDataset = []
imageDataset = []
faultDataset = []

count=1

#final_path = os.path.join

for percent in percentageShort:
    count=1
    percent=str(percent)
    #print(percent)
    for sect in section:
        for k,winDing in enumerate(winding):
            windingB = windingBool[k]
            for j, pHase in enumerate(phase):
                phaseB = phaseBool[j]
                locationBool = [phaseB + windingB]
                
                if j==0:
                    percentBool = np.array([float(percent),0,0])
                elif j==1:
                    percentBool = np.array([0,float(percent),0])
                elif j==2:
                    percentBool = np.array([0,0,float(percent)])
                
                for i,param in enumerate(parameter):
                    new_path = os.path.join(default_path,pHase,winDing,sect,param,(name[i]+percent+ext))
                    #print(new_path)
                    
                    if count<=6 and count>0:
                        dat = [new_path,locationBool,percentBool,faultBool[1]]
                        dataset1.append(dat)
                    if count<=12 and count>6:
                        dat = [new_path,locationBool,percentBool,faultBool[1]]
                        dataset2.append(dat)
                    if count<=18 and count>12:
                        dat = [new_path,locationBool,percentBool,faultBool[1]]
                        dataset3.append(dat)
                    if count<=24 and count>18:
                        dat = [new_path,locationBool,percentBool,faultBool[1]]
                        dataset4.append(dat)
                    if count<=30 and count>24:
                        dat = [new_path,locationBool,percentBool,faultBool[1]]
                        dataset5.append(dat)
                        dat = [new_path,locationBool,percentBool,faultBool[1]]
                    if count<=36 and count>30:
                        dat = [new_path,locationBool,percentBool,faultBool[1]]
                        dataset6.append(dat)
                    
                    count= count+1

datasetPrime = dataset1 + dataset2 +dataset3 + dataset4 + dataset5 + dataset6
i_count = np.arange(0,len(datasetPrime),2)


# %%
batch_size = 1             
num_images = 6
percentage_class = 1
location_class = 3
length = int((1200 * num_images))
num_data = int(length/6)
print(f'{num_data} sets of images will be generated.')
train_split = 0.80
channels = num_images * 3 #6 RGB images (3 channels)
imageDataset = numpy.ndarray(shape=(288, 432, channels))
num_train = int(train_split * num_data)
num_val = int(0.7*(num_data - num_train))
num_test = (num_data - num_train) - num_val
steps_per_epoch = int((num_train/batch_size)/15)
train_validation_steps = int(num_val/batch_size)
test_validation_steps = int(num_test/batch_size)
print('num_train:',num_train, 'num_val:',num_val, 'num_test:',num_test)
print('steps_per_epoch:', steps_per_epoch)
print('train_validation_steps:', train_validation_steps, 'test_validation_steps:', test_validation_steps)
count = 0

# %% [markdown]
# range((800*6),(800*6)+(800*6))
# %% [markdown]
# count=800

# %%
short_dat=[]
im_data = []
image_dat=[]
loc_data =[]  
    
for i in range(length):
        #print(i)
        if int(i)%6 == 0:
            short = pShort[count]
            #print('short',short)
            count+=1
        lenShort = len(str(short))    
        if lenShort<=4:
            image_path = (datasetPrime[i][0])[:-9]
            param_type = (datasetPrime[i][0])[-9]
            short_percent = (datasetPrime[i][0])[-8:-4]
            #print(short_percent, image_path)
            #break
        else:
            image_path = (datasetPrime[i][0])[:-9-1]
            param_type = (datasetPrime[i][0])[-9-1]
            short_percent = (datasetPrime[i][0])[-9:-4]
            #print(short_percent, image_path)
            #break      
        #print(str(image_path + (f'*{param_type}{short}*.png')))    
        files=glob.glob(str(image_path + (f'*{param_type}{short}*.png')))
        flen = len(files)
        
        if flen != 3:
            for file in files:
                #os.remove(file)
                raise Exception(f'Wrong file count: Expected 3 files but got {flen} {file}. Please rectify')
           

            
        if i%2 is 1:
            for myFile in files:
                #print(myFile)
                image=Image.open(myFile)
                image = cv2.imread (myFile)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.reshape(image.shape[0],image.shape[1],image.shape[2])
                #plt.imshow(image)
                #plt.show()
                if image.shape != (288, 432, 3):
                    print(myFile)
                    plt.imshow(image)
                    plt.show()
                    print(image.shape)
                    #os.remove(myFile)
                    prepData()
                    break
                im_data.append(np.asnumpy(image))
                proceed = True
            #print(count)
    
        else:
            for myFile in files:
                #print(myFile)
                image=Image.open(myFile)
                image = cv2.imread (myFile)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.reshape(image.shape[0],image.shape[1],image.shape[2])
                #plt.imshow(image)
                #plt.show()
                if image.shape != (288, 432, 3):
                    print(myFile)
                    plt.imshow(image)
                    plt.show()
                    print(image.shape)
                    #os.remove(myFile)
                    prepData()
                    break
                im_data.append (np.asnumpy(image))
                proceed = False
            
        if proceed:
 
            imageDataset = numpy.concatenate((im_data[0], im_data[1],
                                              im_data[2], im_data[3],
                                              im_data[4], im_data[5]), axis=2)
            image_dat.append(imageDataset)
            value = (datasetPrime[i][1][0])
            if value=='10010' or value=='10001':
               value = 0
            elif value=='01010' or value=='01001':
                value = 1
            elif value=='00110' or value=='00101' :
                value = 2
            
            loc_data.append(value)
            #print('image_data shape:', np.asnumpy(image_dat).shape)           
            #print('im_data shape:', np.array(im_data).shape)
            short_dat.append(short)
            #print('short',short)
            #print('count',count)
            im_data = []

short_data = np.asnumpy(short_dat)/100.         
image_dat = np.asnumpy(image_dat)/255.
y_data=[]
for i in range(len(short_data)):
    y_data.append([short_data[i],loc_data[i]])
print('imageDataset shape: ', imageDataset.shape)
print('short_data size: ', short_data.shape)
print(len(image_dat), 'sets of images prepared')

x_train, x_test, y_train, y_test = train_test_split(image_dat, y_data, train_size=train_split, random_state=50)
#x_train=image_dat[:num_train]
#x_test=image_dat[num_train:]
#y_train=y_data[:num_train]
#y_test=y_data[num_train:]


del image_dat
gc.collect()
y_train = np.asnumpy(y_train)
y_test = np.asnumpy(y_test)

print('x_train shape: ', x_train.shape) 
print('y_train[0] shape: ', y_train[:,0].shape)
print('y_train[1] shape: ', y_train[:,1].shape)    
print('x_test shape: ', x_test.shape)  
print('y_test[0] shape: ', y_test[:,0].shape)
print('y_test[1] shape: ', y_test[:,1].shape)

# %% [markdown]
# %store -r x_train
# %store -r x_test
# %store -r y_train
# %store -r y_test
# %% [markdown]
# (samples, time, rows, cols, channels)
# (None , 1, 360, 1080, 3) means that you have only one sample that is a sequence of 1 images.

# %%
def batch_generator(batch_size, train=None, validation=None):
    """
    Generator function for creating random batches of training-data.
    """
    if train:
        num_samples = num_train
        x_samples = x_train
        y_samples = y_train
        print('using train samples')
    elif validation:
        num_samples = num_val
        x_samples = x_test[:num_samples]
        y_samples = y_test[:num_samples]
        print('using validation samples')
    else:
        num_samples = num_test
        x_samples = x_test[-num_samples:]
        y_samples = y_test[-num_samples:]
        print('using test samples')
    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, x_samples.shape[1],x_samples.shape[2],x_samples.shape[3])
        x_batch = numpy.empty(shape=x_shape,)
        #print(x_shape)
        
        # Allocate a new array for the batch of output-signals.
        y1_shape = (batch_size, percentage_class)
        y1_batch = numpy.empty(shape=y1_shape,)
        y2_shape = (batch_size, 1)
        y2_batch = numpy.empty(shape=y2_shape,)
            
        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = int(np.random.randint(num_samples - 1))
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x_samples[idx]

            y1_batch[i] = np.asnumpy(y_samples[:,0][idx])
            y2_batch[i] = np.asnumpy(y_samples[:,1][idx])
            

        x_batch=x_batch.reshape(batch_size, x_samples.shape[1],x_samples.shape[2],x_samples.shape[3])
        y_batch=[y1_batch, to_categorical(y2_batch, num_classes=location_class, dtype='float32')]
        
        #print(y_batch)
        yield (x_batch, y_batch)


# %%
train_generator = batch_generator(batch_size=batch_size, train=True, validation=True)
x_train_batch, y_train_batch=next(train_generator)

print('x_train shape: ', x_train_batch.shape, 'x_train dtype:', x_train_batch.dtype)  
print('y_train[0] shape: ', y_train_batch[0].shape, 'y_train[0] dtype:', y_train_batch[0].dtype)
print('y_train[1] shape: ', y_train_batch[1].shape, 'y_train[1] dtype:', y_train_batch[1].dtype)


# %%
val_generator = batch_generator(batch_size=batch_size, train=False, validation=True)
x_val_batch, y_val_batch=next(val_generator)

print('x_val shape: ', x_val_batch.shape, 'x_val dtype:', x_val_batch.dtype)  
print('y_val[0] shape: ', y_val_batch[0].shape, 'y_val[] dtype:', y_val_batch[0].dtype)
print('y_val[1] shape: ', y_val_batch[1].shape, 'y_val[] dtype:', y_val_batch[1].dtype)


# %%
test_generator = batch_generator(batch_size=batch_size, train=False, validation=False)
x_test_batch, y_test_batch=next(test_generator)

print('x_test shape: ', x_test_batch.shape, 'x_test dtype:', x_test_batch.dtype)  
print('y_test[0] shape: ', y_test_batch[0].shape, 'y_test dtype:', y_test_batch[0].dtype)
print('y_test[1] shape: ', y_test_batch[1].shape, 'y_test dtype:', y_test_batch[1].dtype)

# %% [markdown]
# #Prepare Validation data
# 
# x_val_shape = (len(x_test), sequence_length, x_train.shape[1], x_train.shape[2], x_train.shape[3])
# x_val = numpy.empty(shape=x_val_shape)
# 
# for i in range(len(x_test)):
#     x_val[i] = x_test[i]
# 
# x_val = x_val.reshape(len(x_test), sequence_length, x_train.shape[1], x_train.shape[2],x_train.shape[3])
# 
# validation_data = (x_val[:batch_size], y_test[:batch_size])
#  
# print('x_val shape: ', validation_data[0].shape)  
# print('y_val shape: ', validation_data[1].shape)

# %%
optimizer = RMSprop(lr=1e-5)
momentum=0.25
datshape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
datshape

# %% [markdown]
# tf.keras.backend.clear_session()
# gc.collect()
# del model
# model = Sequential()
# 
# #First Convolutional layer
# model.add(Conv2D(filters = 64, kernel_size = (3,3), input_shape = datshape, padding='same', activation='selu', data_format='channels_last'))
# model.add(MaxPooling2D(pool_size = (2,2)))
# #model.add(BatchNormalization())
# #model.add(Dropout(0.05))
# 
# #Second Convolutional layer
# model.add(Conv2D(filters = 64, kernel_size = (3,3), padding='same', activation='selu', data_format='channels_last'))
# model.add(MaxPooling2D(pool_size = (2,2)))
# #model.add(BatchNormalization())
# #model.add(Dropout(0.05))
# 
# #Third Convolutional layer
# model.add(Conv2D(filters = 128, kernel_size = (3,3), padding='same', activation='selu', data_format='channels_last'))
# model.add(MaxPooling2D(pool_size = (2,2)))
# #model.add(BatchNormalization())
# #model.add(Dropout(0.05))
# 
# #Fourth Convolutional layer
# model.add(Conv2D(filters = 128, kernel_size = (3,3), padding='same', activation='selu', data_format='channels_last'))
# model.add(MaxPooling2D(pool_size = (2,2)))
# #model.add(BatchNormalization())
# #model.add(Dropout(0.05))
# 
# #Fifth Convolutional layer
# model.add(Conv2D(filters = 256, kernel_size = (3,3), padding='same', activation='selu', data_format='channels_last'))
# model.add(MaxPooling2D(pool_size = (2,2)))
# #model.add(BatchNormalization())
# #model.add(Dropout(0.1))
# 
# #Sixth Convolutional layer
# model.add(Conv2D(filters = 256, kernel_size = (3,3), padding='same', activation='selu', data_format='channels_last'))
# model.add(MaxPooling2D(pool_size = (2,2)))
# #model.add(BatchNormalization())
# #model.add(Dropout(0.1))
# 
# #Seventh Convolutional layer
# model.add(Conv2D(filters = 256, kernel_size = (3,3), padding='same', activation='selu', data_format='channels_last'))
# model.add(MaxPooling2D(pool_size = (2,2)))
# #model.add(BatchNormalization())
# #model.add(Dropout(0.1))
# 
# #Eighth Convolutional layer
# model.add(Conv2D(filters = 512, kernel_size = (3,3), padding='same', activation='selu', data_format='channels_last'))
# model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))
# #model.add(BatchNormalization())
# #model.add(Dropout(0.1))
# 
# 
# #Flattening
# #model.add(BatchNormalization())
# model.add(Flatten())
# 
# 
# #Hidden Layer
# #model.add(Dense(units = 1024, activation='selu')) 
# #model.add(Dropout(0.5))
# #model.add(BatchNormalization())
# 
# #Hidden Layer
# model.add(Dense(units=4096, activation='selu'))
# model.add(Dropout(0.25))
# #model.add(BatchNormalization())
# 
# #Hidden Layer
# model.add(Dense(units=4096, activation='selu'))
# model.add(Dropout(0.25))
# #model.add(BatchNormalization())
# 
# #Output Layer
# model.add(Dense(percentage_class, activation='selu'))
# 
# 
# model.compile(loss='mse', optimizer=optimizer)
# 
# model.summary()

# %%
tf.keras.backend.clear_session()
gc.collect()
#del model

def create_convnet(img_path='Interturn_AI_model_image.png'):
  
    image_input = Input(shape = datshape, name="3-phase_Scaleograms_input")
    
    first_Conv2D = Conv2D(filters = 64, kernel_size = (3,3), input_shape = datshape, padding='same', activation='selu', data_format='channels_last')(image_input)
    first_Pooling = MaxPooling2D(pool_size = (2,2))(first_Conv2D)
    
    second_Conv2D = Conv2D(filters = 64, kernel_size = (3,3), padding='same', activation='selu', data_format='channels_last')(first_Pooling)
    second_Pooling = MaxPooling2D(pool_size = (2,2))(second_Conv2D)
    
    third_Conv2D = Conv2D(filters = 82, kernel_size = (3,3), padding='same', activation='selu', data_format='channels_last')(second_Pooling)
    third_Pooling = MaxPooling2D(pool_size = (2,2))(third_Conv2D)
    
    forth_Conv2D = Conv2D(filters = 82, kernel_size = (3,3), padding='same', activation='selu', data_format='channels_last')(third_Pooling)
    forth_Pooling = MaxPooling2D(pool_size = (2,2))(forth_Conv2D)
    
    fifth_Conv2D = Conv2D(filters = 128, kernel_size = (3,3), padding='same', activation='selu', data_format='channels_last')(forth_Pooling)
    fifth_Pooling = MaxPooling2D(pool_size = (2,2))(fifth_Conv2D)
    
    sixth_Conv2D = Conv2D(filters = 128, kernel_size = (3,3), padding='same', activation='selu', data_format='channels_last')(fifth_Pooling)
    sixth_Pooling = MaxPooling2D(pool_size = (2,2))(sixth_Conv2D)
    
    seventh_Conv2D = Conv2D(filters = 256, kernel_size = (3,3), padding='same', activation='selu', data_format='channels_last')(sixth_Pooling)
    seventh_Pooling = MaxPooling2D(pool_size = (2,2))(seventh_Conv2D)
    
    eighth_Conv2D = Conv2D(filters = 512, kernel_size = (3,3), padding='same', activation='selu', data_format='channels_last')(seventh_Pooling)
    eighth_Pooling = MaxPooling2D(pool_size = (2,2), strides=(2,2))(eighth_Conv2D)
      
    flattened_layer = Flatten()(eighth_Pooling)
    
    first_DNN_0 =  Dense(units=4096, activation='selu')(flattened_layer)
    first_Dropout_0 = Dropout(0.25)(first_DNN_0)

    second_DNN_0 =  Dense(units=4096, activation='selu')(first_Dropout_0)
    second_Dropout_0 = Dropout(0.25)(second_DNN_0)

    first_DNN_1 =  Dense(units=256, activation='selu')(second_Dropout_0)
    first_Dropout_1 = Dropout(0.25)(first_DNN_1)

    second_DNN_1 =  Dense(units=256, activation='selu')(first_Dropout_1)
    second_Dropout_1 = Dropout(0.25)(second_DNN_1)
    out_1 =  Dense(percentage_class, activation='selu', name='Magnitude')(second_Dropout_0)
    
    
    out_2 =  Dense(location_class, activation='sigmoid', name='Location')(second_Dropout_1)

        
    model = Model(inputs=image_input, outputs=[out_1, out_2], name='Inter-turn_Fault_Detection_AI_model')
    
    return model


model = create_convnet()

model.compile(loss=['mean_squared_error','categorical_crossentropy'], optimizer=optimizer,)

model.summary()

# %% [markdown]
# ### Callback Functions
# 
# During training we want to save checkpoints and log the progress to TensorBoard so we create the appropriate callbacks for Keras.
# %% [markdown]
# 
# This is the callback for writing checkpoints during training.

# %%
path_checkpoint = r'C:\Users\hp\iCloudDrive\Final Year Project\Python Stuff\AI model\Model_checkpoint\Inter-turn_fault_detect_model_checkpoint_6.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

# %% [markdown]
# This is the callback for stopping the optimization when performance worsens on the validation-set.

# %%
callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=100, verbose=1)

# %% [markdown]
# This is the callback for writing the TensorBoard log during training.

# %%
folderNumber=3
folderCount = str(3)  
#folderCount=str(input("Enter the Session RUN number: "))
NAME = 'run'+ folderCount
logdir = os.path.join(r'logs', NAME)
Address=str(os.path.join(r'E:\Dropbox\AI' ,logdir))
#if os.path.exists(Address):
    #os.remove(Address)
    #NAME = 'run'+ (folderCount+1)
    #logdir = os.path.join(r'logs', NAME)
    #Address=str(os.path.join(r'E:\Dropbox\AI' ,logdir))
    #if os.path.exists(Address):
    #raise Exception('folder exists')

print(Address)
callback_tensorboard = TensorBoard(log_dir=Address,
                                   histogram_freq=0,
                                   write_graph=True,
                                   write_images=True)

# %% [markdown]
# This callback reduces the learning-rate for the optimizer if the validation-loss has not improved since the last epoch (as indicated by `patience=0`). The learning-rate will be reduced by multiplying it with the given factor. We set a start learning-rate of 1e-3 above, so multiplying it by 0.1 gives a learning-rate of 1e-4. We don't want the learning-rate to go any lower than this.

# %%

callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-6,
                                       patience=0,
                                       verbose=1)


# %%
callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]


# %%
filepath = r'C:\Users\hp\iCloudDrive\Final Year Project\Python Stuff\AI model\Model_architecture\Inter-turn_Fault_Detection_AI_model_6'
def train_model(resume, epochs, initial_epoch, batch_size,model):
    def fit_model():
        
        print(model.summary())
        history=model.fit(    train_generator, 
                              steps_per_epoch=steps_per_epoch, 
                              epochs=EPOCHS, 
                              verbose=1, 
                              callbacks=callbacks,
                              validation_data=val_generator, 
                              validation_steps=train_validation_steps, 
                              #validation_freq=1,
                              #class_weight=None, 
                              #max_queue_size=10, 
                              #workers=8, 
                              #use_multiprocessing=False,
                              shuffle=True,) 
                              #initial_epoch=initial_epoch)
        model.load_weights(path_checkpoint)            
        model.save(filepath)
        model.evaluate(test_generator, steps=test_validation_steps)
        
        return history
    
    if resume:
        try:
            #del model
            #model = load_model(filepath)
            model.load_weights(path_checkpoint)
            print(model.summary())
            print("Model loading....")
            model.evaluate(test_generator, steps=test_validation_steps)
            
        except Exception as error:
            print("Error trying to load checkpoint.")
            print(error)

            
    def plot_train_history(history, title):
        loss = history.history['loss']
        val_loss = np.asnumpy(history.history['val_loss'])
        accuracy = 1-np.asnumpy(loss)
        val_accuracy = 1-np.asnumpy(val_loss)
        epochs = range(len(loss))
        plt.figure(figsize=(15,5))
        plt.plot(epochs, loss, label='training_loss') 
        plt.plot(epochs, val_loss, label='validation_loss')
        plt.title(title)
        plt.legend()
        plt.show()
        plt.figure(figsize=(15,5))
        plt.plot(epochs, accuracy, label='training_accuracy') 
        plt.plot(epochs, val_accuracy, label='validation_accuracy')
        plt.title(title)
        plt.legend()
        plt.show()
        
    # Training the Model
    history = fit_model()
    plot_train_history(history, 'Model Training History ')
    return


# %%
EPOCHS=2000
#steps_per_epoch = int((num_train/batch_size)/17)
train_model(resume=False, epochs=EPOCHS, initial_epoch=0, batch_size=batch_size, model=model)


# %%
model.load_weights(path_checkpoint)
model.evaluate(train_generator, steps=train_validation_steps)
model.evaluate(test_generator, steps=test_validation_steps)


# %%
y_pred = pd.DataFrame(model.predict(x_test))
y_pred = np.asnumpy(y_pred).reshape(-1,2)
y_pred = [np.asnumpy(list(y_pred[:,0])).reshape(-1,1) , np.asnumpy(list(y_pred[:,1])).reshape(-1,3)]
y_pred[1] = y_pred[1].argmax(axis=-1)
y_pred = [y_pred[0].reshape(len(y_pred[0]),1), y_pred[1].reshape(len(y_pred[1]),1)]

y_pred_shape = (len(y_pred[1]), 1)
y_pred_0 = numpy.empty(shape=y_pred_shape,)
y_pred_1 = numpy.empty(shape=y_pred_shape, dtype='int32')
            
# Fill the batch with random sequences of data.
for i in range(len(y_pred[1])):
    y_pred_0[i] = np.asnumpy(y_pred[0][i])
    y_pred_1[i] = np.asnumpy(y_pred[1][i])

Y_pred = numpy.concatenate([y_pred_0, y_pred_1], axis=1)
Y_pred.shape


# %%
y_true = y_test
y_true.shape


# %%
#style.use('classic')
title='Test Set Evaluation'
def plot_prediction():
        plt.figure(figsize=(30,10))
        plt.plot( y_true[:,0]*100, 'o', label='True values') 
        plt.plot( Y_pred[:,0]*100, 'o',label='Predicted values')
        plt.grid()
        plt.title(title)
        plt.legend()
        plt.show()
        plt.figure(figsize=(30,10))
        plt.plot( y_true[:,1],  label='True values') 
        plt.plot( Y_pred[:,1],  label='Predicted values')
        plt.grid()
        plt.title(title)
        plt.legend()
        plt.show()
        return

plot_prediction()


# %%


