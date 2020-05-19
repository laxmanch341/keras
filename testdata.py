import cv2 
import tensorflow as tf
from keras.models import load_model

CATEGORIES=['Dog','Cat']
image='/home/testimages/1001.jpg'


def prepare(image):
    img_size=100
    img_array=cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    new_array=cv2.resize(img_array,(img_size,img_size))
    return new_array.reshape(-1,img_size,img_size,1)


model = tf.keras.models.load_model("/home/Dogs_vs_Cats_CNN.model")
prediction=model.predict([prepare(image)])
print(CATEGORIES[int(prediction[0][0])])


  
