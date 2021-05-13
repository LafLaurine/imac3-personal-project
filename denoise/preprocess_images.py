import os
import cv2
import numpy as np
 

for root, dirs, files in os.walk('../dataset/FlickerImages'):
  for file in files[:500]:
    img = cv2.imread(root + "/" + file)
    gauss = np.random.normal(0,1,img.size)
    gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
    noise = img + img * gauss
    cv2.imwrite("../dataset/FlickerNoisyImages/"+file,noise)