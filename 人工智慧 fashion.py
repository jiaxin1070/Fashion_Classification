# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 20:25:21 2022

@author: user
"""

import os
import numpy as np
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#讀取train資料
path = 'D:\\ai picture train'
files = os.listdir(path)
images = []
label=[]
for f in files:
    path_path = path +'\\'+f
    pic= os.listdir(path_path)
    for p in pic:
       img_path = path +'\\'+f+'\\'+p
       img = image.load_img(img_path,target_size=(28, 28))
       img_array = image.img_to_array(img)
       images.append(img_array)
       lab= p.split(".")[0]
       labs=(int(lab)/100-1)*0.5
       labs=int(labs)
       label.append(labs)
train_x= np.array(images)
train_y=np.array(label)

#讀取test資料
path1 = 'D:\\ai picture test'
files1= os.listdir(path)
images1 = []
label1=[]
for f in files1:
    path_path = path1+'\\'+f
    pic= os.listdir(path_path)
    for p in pic:
       img_path = path1 +'\\'+f+'\\'+p
       img = image.load_img(img_path,target_size=(28, 28))
       img_array = image.img_to_array(img)
       images1.append(img_array)
       lab= p.split(".")[0]
       labs=(int(lab)/100-1)*0.5
       labs=int(labs)
       label1.append(labs)
test_x= np.array(images1)
test_y=np.array(label1)

x_train_n=np.array(train_x)
x_train_m=x_train_n/255

x_val_n=np.array(test_x)
x_val_m=x_val_n/255

y_train_n=np.array(train_y)
y_val_n=np.array(test_y)

#model
from keras.models import Sequential, model_from_yaml, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.optimizers import adam_v2
from tensorflow.keras.optimizers import SGD

model = Sequential()
model.add(Conv2D(16,(3,3), activation = 'relu', input_shape = x_train_m.shape[1:]))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(16,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer="sgd",loss='sparse_categorical_crossentropy',metrics=['accuracy'])

hist=model.fit(x_train_m, y_train_n, epochs=10, batch_size=2,validation_data=(x_val_m, y_val_n))

#畫圖
plt.plot(hist.history['accuracy'], 'b', label='train')
plt.plot(hist.history['val_accuracy'], 'r', label='valid')
plt.legend()
plt.grid(True)
plt.title('accuracy')
plt.show()

plt.plot(hist.history['loss'], 'b', label='train')
plt.plot(hist.history['val_loss'], 'r', label='valid')
plt.legend()
plt.grid(True)
plt.title('loss')
plt.show()

train_set=[0,0.5,1]
#test
test_image = image.load_img('D:\\test.JPG', target_size = (28, 28))
test_img = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
if result[0][0]==0.:
    prediction= 'japan'


elif result[0][0]==0.5:
    prediction= 'korean'


elif result[0][0]==1.:
    prediction= 'usa'

