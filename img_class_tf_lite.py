import numpy as np
from PIL import Image
import tensorflow as tf # TF2
import cv2

interpreter = tf.lite.Interpreter("model_v2.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = 150
path = "./data/unknown/4.jpg"
label = "unknown"
img = cv2.imread(path)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

input_data = np.array(img, dtype=np.float32)
print (input_data.shape)


input_data = np.expand_dims(input_data, 0)
print (input_data.shape)
input_data = input_data/255
#print (input_data[0])

interpreter.set_tensor(input_details[0]["index"], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print ("green\topen\tyellow")
list = output_data.tolist()
for i in range(len(list)):
    for j in list[i]:
        print (round((j * 100),), end =" " + "%\t")
