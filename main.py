import numpy as np
import os
import split_ucf11 as su
import pandas as pd
from keras.layers import Input,LSTM,GlobalAveragePooling2D, Dense, Conv2D,Conv2DTranspose, Activation, Reshape,Flatten, BatchNormalization,Dropout,MaxPooling2D,TimeDistributed, CuDNNLSTM
from keras import regularizers
from keras.models import Model, Sequential, load_model
from keras import optimizers
from moviepy.editor import VideoFileClip
from scipy.io import savemat, loadmat
from keras.utils import to_categorical
import matplotlib.pyplot as plt


'''Data Preprocessing'''
##Convert all videos to same resolution and save as mp4 file
def resize_and_save_to_mp4():
    groups = su.load_groups('UCF11_updated_mpg')
    video_paths = []
    for i in range(len(groups)):
        for file in os.listdir(groups[i][0]):
            if (os.path.splitext(file)[1] == '.mpg'):
                video_paths.append(os.path.join(groups[i][0],file))
    for index in range(len(video_paths)):
        clip = VideoFileClip(video_paths[index])
        clip = clip.resize((224,224))
        path = os.path.splitext(video_paths[index])[0]
        ext = '.mp4'
        clip.write_videofile(path+ext)
    print("Resize Done")

#split into train and test and save as csv file    
def split_train_test_map():
    groups = su.load_groups('UCF11_updated_mpg')
    train,test = su.split_data(groups, '.mp4')
    su.write_to_csv(train,'train_map_mp4.csv')
    su.write_to_csv(test, 'test_map_mp4.csv')
    
##Extract 1 Frame, save into matrix, X = [N,224,224,3]
def save_1frame_to_mat():
    train = pd.read_csv('train_map_mp4.csv', header=None, names = ['Path','Label'])
    x_train_1frame = np.zeros((train.shape[0],224,224,3)).astype('uint8')
    for i in range(train.shape[0]):
        clip = VideoFileClip(train['Path'][i])
        x_train_1frame[i] = clip.get_frame(0)
    test = pd.read_csv('test_map_mp4.csv', header=None, names = ['Path','Label'])
    x_test_1frame = np.zeros((test.shape[0],224,224,3)).astype('uint8')
    for i in range(test.shape[0]):
        clip = VideoFileClip(test['Path'][i])
        x_test_1frame[i] = clip.get_frame(0)
    savemat('x_train_1frame.mat',mdict = {'data':x_train_1frame})
    savemat('x_test_1frame.mat',mdict = {'data':x_test_1frame})

##Extract 5 Frames, save into matrix, X = [N,5,224,224,3]
def save_5frame_to_mat():
    train = pd.read_csv('train_map_mp4.csv', header=None, names = ['Path','Label'])
    x_train_5frame = np.zeros((train.shape[0],5,224,224,3)).astype('uint8')
    for i in range(train.shape[0]):
        clip = VideoFileClip(train['Path'][i])
        totalframes = clip.fps*clip.duration
        increment = totalframes // 5
        for j in range(5):
            x_train_5frame[i][j] = clip.get_frame((j*increment)/clip.fps)
        print("Video" + str(i+1))
    test = pd.read_csv('test_map_mp4.csv', header=None, names = ['Path','Label'])
    x_test_5frame = np.zeros((test.shape[0],5,224,224,3)).astype('uint8')
    for i in range(test.shape[0]):
        clip = VideoFileClip(test['Path'][i])
        totalframes = clip.fps*clip.duration
        increment = totalframes // 5
        for j in range(5):
            x_test_5frame[i][j] = clip.get_frame((j*increment)/clip.fps)
        print("Video" + str(i+1))
    savemat('x_train_5frame.mat',mdict = {'data':x_train_5frame})
    savemat('x_test_5frame.mat',mdict = {'data':x_test_5frame})
    
def save_label_to_mat():
    train = pd.read_csv('train_map_mp4.csv', header=None, names = ['Path','Label'])
    y_train = np.array(train['Label']).astype('uint8')
    y_train = to_categorical(y_train,11)
    test = pd.read_csv('test_map_mp4.csv', header=None, names = ['Path','Label'])
    y_test = np.array(test['Label']).astype('uint8')
    y_test = to_categorical(y_test,11)
    savemat('y_train.mat',mdict = {'data':y_train})
    savemat('y_test.mat',mdict = {'data':y_test})

#resize_and_save_to_mp4()
#split_train_test_map()
#save_1frame_to_mat()
#save_5frame_to_mat()
#save_label_to_mat()


'''1 Frame CNN'''  
x_train_1frame,y_train = loadmat('x_train_1frame.mat')['data'], loadmat('y_train.mat')['data']
x_test_1frame,y_test = loadmat('x_test_1frame.mat')['data'], loadmat('y_test.mat')['data']
#
x_train_1frame, x_test_1frame = x_train_1frame/255, x_test_1frame/255

#
model = Sequential()
model.add(Conv2D(32,2,activation = 'relu',strides= 2, padding = 'same',input_shape= (224,224,3)))
model.add(BatchNormalization())
model.add(Conv2D(64,2,strides= 1,activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128,2,strides= 2,activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(256,2,strides= 1,activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512,2,strides= 2,activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(1024,2,strides= 1,activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.8))
model.add(Dense(1024,activation = 'relu'))
model.add(Dropout(0.8))
model.add(Dense(11,activation = 'softmax'))
model.summary()

opti = optimizers.Adam(lr=0.0001)
model.compile(optimizer = opti, loss = 'categorical_crossentropy', metrics = ['accuracy']) 
history = model.fit(x_train_1frame, y_train, batch_size = 4, epochs = 100, shuffle= True, validation_data=[x_test_1frame,y_test])
model.save_weights('cnn_1frame_weights.h5')
model.save('cnn_1frame_model.h5')
model.evaluate(x_test_1frame,y_test)
np.mean(history.history['val_acc'])
np.mean(history.history['val_loss'])
np.mean(history.history['acc'])
np.mean(history.history['loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Single Frame CNN Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Single Frame CNN Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()


'''5 Frames CNN-LSTM'''
x_train_5frame,y_train =loadmat('x_train_5frame.mat')['data'],loadmat('y_train.mat')['data']
x_test_5frame,y_test =loadmat('x_test_5frame.mat')['data'], loadmat('y_test.mat')['data']
x_train_5frame, x_test_5frame = x_train_5frame/255, x_test_5frame/255
#

model = Sequential()
model.add(TimeDistributed(Conv2D(32,2,activation = 'relu',strides= 2, padding = 'same'),input_shape= (5,224,224,3)))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Conv2D(64,2,strides= 1,activation = 'relu', padding = 'same')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Conv2D(128,2,strides= 2,activation = 'relu', padding = 'same')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Conv2D(256,2,strides= 1,activation = 'relu', padding = 'same')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Conv2D(512,2,strides= 2,activation = 'relu', padding = 'same')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Conv2D(1024,2,strides= 1,activation = 'relu', padding = 'same')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(GlobalAveragePooling2D()))
model.add(CuDNNLSTM(1024, return_sequences= False))
model.add(Dropout(0.9))
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.9)) 
model.add(Dense(11,activation = 'softmax'))

model.summary()
opti = optimizers.Adam(lr=0.0001)
model.compile(optimizer = opti, loss = 'categorical_crossentropy', metrics = ['accuracy']) 
history = model.fit(x_train_5frame, y_train, batch_size = 4,epochs = 100, shuffle= True, validation_data=[x_test_5frame,y_test])
model.evaluate(x_test_5frame,y_test)
#model.save_weights('cnn_lstm_5frame_weights.h5')
#model.save('cnn_lstm_5frame_model.h5')
np.mean(history.history['val_acc'])
np.mean(history.history['val_loss'])
np.mean(history.history['acc'])
np.mean(history.history['loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Five Frame CNN-LSTM Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Five Frame CNN-LSTM Frame CNN Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()






