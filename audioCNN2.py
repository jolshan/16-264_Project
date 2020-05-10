'''
Tried a second method of feature extraction and CNN. This one didn't turn out 
as well, but I spent less time on this, so there may be a bug in the code.
Dataset comes from:
http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html
Labels for emotions come from:
http://www.robots.ox.ac.uk/~vgg/research/cross-modal-emotions/
Code adapted from:

'''


from keras import layers
from keras import models
import keras.backend as K
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import figure
import gc
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np
from os import path
from sklearn.metrics import confusion_matrix
from keras_preprocessing.image import ImageDataGenerator


def create_spectrogram(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  =  name + '.png'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S

def create_spectrogram_test(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  =  name + '.png'
    fig.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S



# Only creates png files of features if a png for a file does not yet exist.
# Only works with labels 1-5 (no 6,7,8)
traindf=pd.read_csv('filesAndLabelsTrain.csv',dtype=str)
testdf=pd.read_csv('filesAndLabelsTest.csv',dtype=str)


Data_dir= traindf["ID"]

for i in range(0,len(Data_dir)//2000):
    j = i * 2000
    for file in Data_dir[j:j+2000]:
        filename,name = file,file
        if not path.exists(filename+'.png'):
            create_spectrogram(filename,name)
    gc.collect()

for file in Data_dir[(len(Data_dir)//2000)*2000:]:
    filename,name = file,file
    if not path.exists(filename+'.png'):
        create_spectrogram(filename,name)
gc.collect()


Test_dir=testdf["ID"]

for i in range(0,len(Test_dir)//1500):
    for file in Test_dir[i:i+1500]:
        filename,name = file,file
        if not path.exists(filename+'.png'):
            create_spectrogram_test(filename,name)
    gc.collect()
for file in Test_dir[(len(Test_dir)//1500)*1500:]:
    filename,name = file,file
    if not path.exists(filename+'.png'):
        create_spectrogram_test(filename,name)
gc.collect()

def append_ext(fn):
    return fn+".png"

traindf["ID"]=traindf["ID"].apply(append_ext)
testdf["ID"]=testdf["ID"].apply(append_ext)


datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)


train_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=None,
    x_col="ID",
    y_col="Class",
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))

valid_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=None,
    x_col="ID",
    y_col="Class",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
model.compile(optimizers.rmsprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()


#Fitting keras model, no test gen for now
STEP_SIZE_TRAIN=train_generator.n//(train_generator.batch_size*100) 
STEP_SIZE_VALID=valid_generator.n//(valid_generator.batch_size*100)
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=150
)
print(model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID
))

eval_generator=datagen.flow_from_dataframe(
    dataframe=testdf,
    directory=None,
    x_col="ID",
    y_col="Class",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode="categorical",
    target_size=(64,64))
print(model.evaluate_generator(generator=eval_generator, steps=STEP_SIZE_VALID
))


test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
    dataframe=testdf,
    directory=None,
    x_col="ID",
    y_col=None,
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(64,64))
STEP_SIZE_TEST=eval_generator.n//eval_generator.batch_size


test_generator.reset()

pred=model.predict_generator(eval_generator,
steps=STEP_SIZE_TEST,
verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)


minlen = min(len(predicted_class_indices), len(eval_generator.classes))

print(confusion_matrix(eval_generator.classes[1:minlen], predicted_class_indices[1:minlen]))
