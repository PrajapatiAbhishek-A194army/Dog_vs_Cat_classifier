import kagglehub
import os
from PIL import Image
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
import os
from glob import glob
import cv2
from  sklearn.model_selection import train_test_split
import numpy as np
path = kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset")
folder_path_cat=path+"/PetImages/Cat"
folder_path_dog=path+"/PetImages/Dog"
images_cat=os.listdir(folder_path_cat)
images_dog=os.listdir(folder_path_dog)
images=images_cat+images_dog
print(images_cat[:5])
print(images_dog[:5])

cat_dir=glob(folder_path_cat+"/*.jpg")[:2000]
dog_dir=glob(folder_path_dog+"/*.jpg")[:2000]
data=[]
labels=[]
for path in cat_dir:
  img=cv2.imread(path)
  if img is None:
        print("Failed:", path)
        continue
  img=cv2.resize(img,(128,128))

  data.append(img)
  labels.append(0)

for path in dog_dir:
  img=cv2.imread(path)
  if img is None:
        print("Failed:", path)
        continue
  img=cv2.resize(img,(128,128))

  data.append(img)
  labels.append(1)

model=Sequential()

model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

x=np.array(data)/255.
y=np.array(labels)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.2)

history=model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))

model.save("dog_cat_model.h5")