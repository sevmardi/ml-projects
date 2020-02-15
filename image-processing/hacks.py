import cv2
import numpy as np
import time
# from imutils import paths
import imutils

def noisy(image):
    row, col, ch = image.shape
    s_vs_p = 0.5
    amount = 0.5
    out = image
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    out[coords] = 0
    return out

image = cv2.imread('images/img.png', cv2.COLOR_BGR2GRAY)
image = imutils.resize(image, width=min(400, image.shape[1]))

orig = image.copy()

# sub_img = image[y:y + h, x:x + w]
# im = cv2.imread('test.jpg', cv2.COLOR_BGR2GRAY)
x = 1580
y = 277
h = 366
w = 150
# crop_img = im[y:y + h, x:x + w]
# noisy(crop_img)

# rects = cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 0), 2) 
cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
c = 255 * np.random.randint(low=0, high=2, size=(h,w), dtype="i")
image[y:(y+h),x:(x+w),0] = c
image[y:(y+h),x:(x+w),1] = c
image[y:(y+h),x:(x+w),2] = c
cv2.imwrite('sp_noise.jpg', image)
# for x in rects:
#     print (x,y,w,h)
# cv2.imwrite('exp.jpg', image)
