import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from matplotlib.path import Path
import numpy as np
import cv2

lena = mpimg.imread('images/2_resize.jpg')


a = np.array([[[236,528],[200,426],[1000,361],[1100,450],[236,528]]], dtype = np.int32)# to draw the shape
b = Path([(236, 528), (200,426), (1000,361), (1100,450),(236,528)])  # to judge if the point in the target zone
judge_result=b.contains_points([(746, 478)])
if(judge_result[0]==True):
    print('yes')
else:
    print('no')
#cv2.polylines(lena, a, 1, 255)
cv2.fillPoly(lena, a, 255)

plt.imshow(lena)
plt.axis('on')

plt.savefig('images/4_.jpg')
plt.show(10000) 

