from getoptions import GetOptions
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import datetime
import os, glob
import csv

### arg
args = GetOptions()
task_name = args.task
gpu_id = args.gpu
num = args.number
name = args.name
"""
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
"""

### path
traial_name = task_name+'_'+name+'_'+str(num)
train_path = '../data/pretext/'+task_name+'/train/'
valid_path = '../data/pretext/'+task_name+'/valid/'
model_name = '../model/pretext_'+traial_name
fig_name = '../result/pretext_'+traial_name
log_path = '../log/'
log_dir = log_path+'pretext/'+traial_name

### hyper-parameter
epochs = 200
batch_size = 128
if task_name == 'area_4class' or task_name == 'ratio_4class':
    n_categories = 4
elif task_name == 'area_8class' or task_name == 'ratio_8class':
    n_categories = 8
else:
    n_categories = 1

### 
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
if not os.path.exists(log_dir+'/train/'):
    os.mkdir(log_dir+'/train/')
    os.mkdir(log_dir+'/valid/')

### model
base_model=VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
prediction = Dense(n_categories, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=prediction)


model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
model.summary()

callbacks = [
    #EarlyStopping(patience=10),
    #ModelCheckpoint(filepath=model_name+str(num)+'.{epoch:02d}.hdf5', monitor='val_loss', verbose=1, save_best_only=True),
    TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=batch_size)
]

# save model
json_string = model.to_json()
open(model_name+'.json', 'w').write(json_string)


### fit
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True
)
test_datagen = ImageDataGenerator(
    rescale=1./255,
)
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=batch_size
)
valid_generator = test_datagen.flow_from_directory(
    valid_path,
    target_size=(224,224),
    batch_size=batch_size
)

hist = model.fit(
    train_generator, 
    epochs=epochs, 
    steps_per_epoch=len(train_generator), 
    validation_steps=len(valid_generator),
    verbose=1, 
    validation_data=valid_generator,
    callbacks=callbacks
)

### save weights
model.save(model_name+'.h5')


### plot
plt.plot(range(1, epochs+1), hist.history['accuracy'], label='train')
plt.plot(range(1, epochs+1), hist.history['val_accuracy'], label='valid')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(fig_name+'.pdf')

fin_acc = model.evaluate(valid_generator)[1]

with open('../result/pretext.csv','a') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow( (traial_name, str(fin_acc)) )