from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

im = Image.open("/mrtstorage/users/chli/real_data/gt_image2/1571220711.743864059.png")
instances = np.unique(np.array(im))
kernel = np.ones((5, 5), np.uint8)
local_instance_mask = np.array(im) == 0

erosion2 = cv2.erode(local_instance_mask.astype('uint8'), kernel, iterations=1)
dilation2 = cv2.dilate(erosion2, kernel, iterations=1)
boundry = (dilation2 - erosion2) * 125
Img = Image.fromarray(boundry)
Img.show()

#Image.fromarray(boundry.astype('uint8')).show()


'''

#export PYTHONPATH=/home/chli/cc_code2/deeplab/env/lib/python3.6/site-packages


img2 = np.load('./opencv_image/1.npy')
img = cv2.imread('./opencv_image/j.png',0)
print("img2", type(img2[1][1]))
print(img2.shape)
print("img", type(img[1][1]))
print(img.shape)


plt.figure()
# plt.imshow(image)
plt.imshow(img2)
plt.show()

kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img2,kernel,iterations = 1)

plt.figure()
# plt.imshow(image)
plt.imshow(erosion)
plt.show()

dilation = cv2.dilate(erosion,kernel,iterations = 1)


plt.figure()
# plt.imshow(image)
plt.imshow(dilation)
plt.show()

opening = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)

plt.figure()
# plt.imshow(image)
plt.imshow(opening)
plt.show()

closing = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)

plt.figure()
# plt.imshow(image)
plt.imshow(closing)
plt.show()

'''