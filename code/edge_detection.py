# importing libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# reading image
img_2 = cv2.imread('pallet-block-502/pallet-block-502/camera-1-light-on/cropped_145_C.jpg')

# resized all images to 384*128
resized = cv2.resize(img_2, [384, 128], interpolation = cv2.INTER_AREA)

# convert to gray scale and blur the image for better edge detection
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(gray, (5,5), 100)

# applying canny edge detection
edges = cv2.Canny(img_blur, 50, 80)

# showing edged image
# cv2.imshow('Edges', edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# funtion for geting threshold for RL images
def threshold_rl(values):
    rl_1 = np.argmax(values[5:100]) + 5
    rl_2 = np.min(np.argsort(values[250:-5])[-5:]) + 250 #[::-1][:5]
    return rl_1, rl_2
  
rl_1, rl_2 = threshold_rl(np.sum(edges, axis = 0))

plt.plot(np.sum(edges, axis = 0))
plt.axvline(rl_1)
plt.axvline(rl_2)

# function for getting threshold for RR images
def threshold_rr(values):
    rr_1 = np.argmax(np.argsort(values[:100])[-2:])
    rr_2 = np.argmax(values[300:]) + 300
    return rr_1, rr_2

rr_1, rr_2 = threshold_rr(np.sum(edges, axis = 0))
plt.plot(np.sum(edges, axis = 0))
plt.axvline(rr_1)
plt.axvline(rr_2)

# funtion for geting threshold for Left images
def threshold_left(values):
    l_1 = np.argmin(np.argsort(values[0:100]))  
    l_2 = np.argmax(values[300:-5]) + 300 #[::-1][:5]
    return l_1, l_2

l_1, l_2 = threshold_left(np.sum(edges, axis = 0))

plt.plot(np.sum(edges, axis = 0))
plt.axvline(l_1)
plt.axvline(l_2)

# funtion for geting threshold for Right images
def threshold_right(values):
    r_1 = np.argmax(values[0:100])  
    r_2 = np.max(np.argsort(values[250:])) + 250 #[::-1][:5]
    return r_1, r_2

r_1, r_2 = threshold_right(np.sum(edges, axis = 0))

plt.plot(np.sum(edges, axis = 0))
plt.axvline(r_1)
plt.axvline(r_2)


# function fot geting threshold for Center images
def threshold_center(values):
    c_1 = np.argmax(values[20:50]) + 20
    c_2 = np.argmax(values[300:]) + 300
    
    return c_1, c_2

c_1, c_2 = threshold_center(np.sum(edges, axis = 0))  

plt.plot(np.sum(edges, axis = 0))
plt.axvline(c_1)
plt.axvline(c_2)
