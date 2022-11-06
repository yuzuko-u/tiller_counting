#coding:utf-8
from PIL import Image
import sys,keras
import os, glob
import numpy as np
import random, math
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array, load_img
from keras import backend as K
from getoptions import GetOptions
import pandas as pd

import csv

### arg
args = GetOptions()
task_name = args.task
gpu_id = args.gpu
no = args.number
name = args.name
'''
print(tf.__version__)
if tf.__version__ >= "2.1.0":
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)
elif tf.__version__ >= "2.0.0":
    #TF2.0
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(physical_devices[gpu_id], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)
else:
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list=str(gpu_id), # specify GPU number
            allow_growth=True
        )
    )
    set_session(tf.Session(config=config))
'''

epochs=50
n_categories=1
batch_size=32

traial_name = task_name+'_'+name+'_'+str(no)
pretext_model="../model/pretext_"+traial_name
tiller_model = "../model/target_"+traial_name
file_name = '../result/target_ft_'+traial_name


plt.figure(figsize=(7, 7))
plt.rcParams["font.size"] = 18
plt.xlim( [-3, 18] )
plt.xticks( [0, 5, 10, 15] )
plt.ylim( [-3, 18] )
plt.yticks( [0, 5, 10, 15] )
plt.xlabel('Actual values')
plt.ylabel('Predict values')
x = np.arange(-3, 19, 1)
y=x
plt.plot(x, y)

MAE = []

### make dataframe
df = pd.read_csv("../data/target.csv", header=None, names=('id','label'))
df['filename']=df['id'].astype(str)+'.png'

df=df.sample(frac=1)

SAMPLE_SIZE = len(df)
BLOCK_SIZE = SAMPLE_SIZE // 6

preds = [[None]]
for i in range(6):
    INDEX_L = BLOCK_SIZE * i + 1
    INDEX_R = BLOCK_SIZE * (i+1) + 1

    df_train = df.iloc[pd.np.r_[1:INDEX_L, INDEX_R:SAMPLE_SIZE]]
    df_valid = df.iloc[INDEX_L:INDEX_R, :]
            
    #load model and weights
    json_string=open(pretext_model+'.json').read()
    base_model=model_from_json(json_string)
    base_model.load_weights(pretext_model+'.h5')
    
    #add new layers instead of FC networks

    x=base_model.layers[19].output
    x=Dense(512,activation='relu')(x)
    prediction=Dense(n_categories,activation='linear')(x)

    model=Model(inputs=base_model.input, outputs=prediction)
    
    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.003, amsgrad=False),
                loss='mean_squared_error',
                metrics=['mae'])
    
    model.summary()

    callbacks = [
        #EarlyStopping(patience=10),
        #ModelCheckpoint(filepath=model_name+str(num)+'.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True),
        #TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=batch_size)
    ]
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=False,
        vertical_flip=True,
    )

    test_datagen = ImageDataGenerator(
        rescale=1./255,
    )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory='../data/target/',
        x_col='filename',
        y_col='label',
        target_size=(224,224),
        batch_size=batch_size,
        class_mode='raw',
    )

    valid_generator = test_datagen.flow_from_dataframe(
        dataframe=df_valid,
        directory='../data/target/',
        x_col='filename',
        y_col='label',
        target_size=(224,224),
        batch_size=batch_size,
        class_mode='raw',
        shuffle=False
    )

    hist = model.fit(
        train_generator, 
        epochs=epochs, 
        steps_per_epoch=len(train_generator), 
        validation_steps=len(valid_generator),
        verbose=1, 
        validation_data=valid_generator,
        callbacks=callbacks,
    )

    #save model
    json_string=model.to_json()
    open(tiller_model+'.json','w').write(json_string)

    #save weights
    model.save(tiller_model + "_" + str(i+1) +'.h5')

    #load model and weights
    json_string=open(tiller_model+'.json').read()
    model=model_from_json(json_string)
    model.load_weights(tiller_model + "_" + str(i+1) +'.h5')

    model.compile(loss='mean_squared_error', metrics=['mae'])

    valid_generator.reset()
    pred = model.predict(valid_generator)
    mae = model.evaluate(valid_generator)[1]

    MAE.append(mae)
    
    plt.scatter(df_valid['label'], pred, c='r', s=10)

    preds = preds + list(pred)

    '''
    plt.plot(range(1, epochs+1), hist.history['mae'], label='train')
    plt.plot(range(1, epochs+1), hist.history['val_mae'], label='valid')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig('../log/fc'+str(i)+'.pdf')
    '''

plt.savefig(file_name+'.pdf')
txt_file = open(file_name+'.txt', 'w')
for i in range(6):
    txt_file.writelines(str(i+1) + " : " + str(MAE[i]) + "\n")
    print(str(i+1) + " : " + str(MAE[i]))
txt_file.writelines("ave : " + str(sum(MAE)/len(MAE)) + "\n")
txt_file.close() 
print("ave : " + str(sum(MAE)/len(MAE)))

df_result = pd.DataFrame({
    'ground_truth': df['label']
})
df_result['prediction'] = np.array(preds)

df_result.sort_index().to_csv(file_name+'.csv')