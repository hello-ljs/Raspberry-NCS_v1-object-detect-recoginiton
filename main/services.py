import os
import subprocess
import time

yuvfilepath = '/home/pi/workspace/main/tmp_yuv/'
jpgfilepath = '/home/pi/workspace/main/tmp_jpg/'

while 1:
    count = 0
    f_list = []
    for i,j,filename_list in os.walk(yuvfilepath):
        f_list = filename_list
    if f_list:    
        for f in f_list:
            fy = yuvfilepath + f
            nowtime = time.strftime("%Y%m%d%H%M%S", time.localtime())
            fj = jpgfilepath + str(nowtime) + str(count)+ '.jpg'
            cmd1 = './yuv2jpg -i '+ fy + ' -o ' + fj  
            subprocess.call(cmd1, shell=True)
            count += 1
            time.sleep(0.1)
            os.remove(fy)
        time.sleep(3)
        
    else:
        #time.sleep(1)
        continue
    