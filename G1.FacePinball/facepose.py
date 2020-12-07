import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import paddlehub as hub
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np
import math

src_img = cv2.imread('./3.jpg')

shape = src_img.shape
ss = (round(shape[1]/3),round(shape[0]/3))
src_img = cv2.resize(src_img, ss, interpolation=cv2.INTER_AREA)
tmp_img = src_img.copy()
module = hub.Module(name="face_landmark_localization")
result = module.keypoint_detection(images=[src_img])

mask_img = src_img.copy()

tmp = np.array(result[0]['data'][0],dtype=np.int32)

lefteye = tuple([tmp[36][0], tmp[37][1]])
leftsize = tuple([tmp[39][0] - tmp[36][0], tmp[41][1] - tmp[37][1]])
righteye = tuple([tmp[42][0], tmp[43][1]])
leftsize = tuple([tmp[45][0] - tmp[42][0], tmp[47][1] - tmp[43][1]])

bounder_pairs =[[11,14],[14,17],[17,25],[25,20],[20,1],[1,4],[4, 7]]
shape = mask_img.shape
img_mask2 = np.zeros((shape[0], shape[1]), np.uint8)

for i ,(start, end) in enumerate(bounder_pairs,1):
    cv2.line(mask_img, tuple(tmp[start - 1]), tuple(tmp[end - 1]), (0, 0, 255), 3)
    cv2.line(img_mask2, tuple(tmp[start - 1]), tuple(tmp[end - 1]), i, 5)

img_mask2[:200,:100] = 20
print(img_mask2[199,99])

plt.figure(figsize=(10,10))
plt.imshow(img_mask2) 
plt.axis('off') 
plt.show()
plt.close()

mask_pairs = [1,4,7,9,11,14,17,25,20]
shape = mask_img.shape
mask_point = np.array([tmp[x - 1] for x in mask_pairs])
img_mask = np.zeros((shape[0], shape[1]), np.uint8)
cv2.fillPoly(img_mask, [mask_point], 1)

nouse_pairs = [[28,29],[30,31]]
for start, end in nouse_pairs:
    cv2.line(mask_img, tuple(tmp[start - 1]), tuple(tmp[end - 1]), (0, 0, 255), 3)

mask_pairs = [[28,29,29,28],[30,31,31,30]]
mask_point = np.array([[tmp[y - 1][0] + (-2 if i > 1 else 2),tmp[y - 1][1]] for x in mask_pairs for i, y in enumerate(x,0)])
mask_point = np.reshape(mask_point, (2,4,2))
cv2.fillPoly(img_mask, mask_point, 0)

mouth_pairs =[
    [49,50],[50,51],[51,52],[52,53],
    [53,54],[54,55],[55,56],[56,57],
    [57,58],[58,59],[59,60],[60,49]
] 

for start, end in mouth_pairs:
    cv2.line(mask_img, tuple(tmp[start - 1]), tuple(tmp[end - 1]), (0, 0, 255), 3)

mask_pairs = [49,50,51,52,53,54,55,56,57,58,59,60]
mask_point = np.array([tmp[x - 1] for x in mask_pairs])
cv2.fillPoly(img_mask, [mask_point], 0)

plt.figure(figsize=(10,10))
plt.imshow(img_mask) 
plt.axis('off') 
plt.show()

res_img_path = 'face_landmark.jpg'
cv2.imwrite(res_img_path, mask_img)

img = mpimg.imread(res_img_path) 
# 展示预测68个关键点结果
plt.figure(figsize=(10,10))
plt.imshow(img) 
plt.axis('off') 
plt.show()