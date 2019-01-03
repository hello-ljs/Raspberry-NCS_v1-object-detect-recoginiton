import os
#import subprocess
import time

# start capture
os.system("gnome-terminal -e 'bash -c \"cd;cd workspace/live/testProgs/;cktool -s 3 -o test.264; exec bash\"'")
# strat transfer video
os.system("gnome-terminal -e 'bash -c \"cd;cd workspace/live/testProgs/;./testH264VideoStreamer; exec bash\"'")
# start socket.server
os.system("gnome-terminal -e 'bash -c \"cd;cd workspace/main/;python s.py; exec bash\"'")
# if save_image
os.system("gnome-terminal -e 'bash -c \"cd;cd workspace/main/;python services.py; exec bash\"'")
'''
while 1:
    yuvfilepath = '/home/pi/workspace/main/tmp_yuv/out_480_nonfifo.yuv'
    if os.path.isfile(yuvfilepath):
        #os.system("gnome-terminal -e 'bash -c \"cd;./home/pi/workspace/main/test_480; exec bash\"'")
        cmd1 = 'ffmpeg -s 640x480 -i /home/pi/workspace/main/tmp_yuv/out_480_nonfifo.yuv /home/pi/workspace/main/out_allima/yuv_tojpg.png'
        subprocess.call(cmd1, shell=True)
        time.sleep(0.2)
        os.remove(yuvfilepath)
        jpgfilepath = '/home/pi/workspace/main/out_allima/yuv_tojpg.png'
        if os.path.isfile(jpgfilepath):
            #os.system("gnome-terminal -e 'bash -c \"python run_sdk2_cutimage.py;exit; exec bash\"'")
            cmd2 = 'python run_sdk2_cutimage.py'
            subprocess.call(cmd2, shell=True)
            time.sleep(4)
            
        
    else:
        time.sleep(1)
        continue
    
    
if os.path.isfile(jpgfilepath):
    os.system("gnome-terminal -e 'bash -c \"cd ..;python run.py; exec bash\"'")
   
if os.path.isfile(sdkfilepath):
    os.system("gnome-terminal -e 'bash -c \"cd ..;python s.py; exec bash\"'")
'''