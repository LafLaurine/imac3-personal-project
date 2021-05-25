import os
import cv2
import numpy as np

def mat2gray(img):
    A = np.double(img)
    out = np.zeros(A.shape, np.double)
    normalized = cv2.normalize(A, out, 1.0, 0.0, cv2.NORM_MINMAX)
    return out
 
 #Add noise to the image
def random_noise(image, seed=None, clip=True, **kwargs):
    image = mat2gray(image)
    if image.min() < 0:
        low_clip = -1
    else:
        low_clip = 0
    if seed is not None:
        np.random.seed(seed=seed)
        
    noise = np.random.normal(kwargs['mean'], kwargs['var'] ** 0.5,
                                image.shape)        
    out = image  + noise
    if clip:        
        out = np.clip(out, low_clip, 1.0)
    return out 
  
for root, dirs, files in os.walk('../dataset/FlickerImages'):
    files.sort()
    for file in files[:500]:
        img = cv2.imread(root + "/" + file)
        noisy = random_noise(img, mean=0.1,var=0.01)
        noisy = np.uint8(noisy*255)
        cv2.imwrite("../dataset/FlickerNoisyImages/"+file,noisy)