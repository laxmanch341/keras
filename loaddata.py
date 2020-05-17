import numpy as np
import cv2
import os

DataDir= "/home/PetImages"

CATEGORIES=["Dog","Cat"]

for i in CATEGORIES:
    path=os.path.join(DataDir,i)
    print(path)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        break
    break



img_size=100

new_array=cv2.resize(img_array,(100,100))

training_data=[]

def create_training_data():
    for i in CATEGORIES:

        path=os.path.join(DataDir,i)
        class_num=CATEGORIES.index(i)

        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array,(img_size,img_size))
                training_data.append([new_array,class_num])
            
            except Exception as e:
                pass


create_training_data()
print(len(training_data))
            
import random
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample)




X=[]
y=[]

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1,img_size,img_size,1))


X=np.array(X).reshape(-1,img_size,img_size,1)

import pickle

pickle_out=open("/home/X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out=open("/home/y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()

