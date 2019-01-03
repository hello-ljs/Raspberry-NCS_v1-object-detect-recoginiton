from matplotlib.path import Path
import cv2

lena = cv2.imread('images/4.jpg')


b = Path([(200, 0), (2530,2018), (2585,1355), (1737,1591)])  # to judge if the point in the target zone
judge_result=b.contains_points([(2300, 1591)])
if(judge_result[0]==True):
    print('yes')


