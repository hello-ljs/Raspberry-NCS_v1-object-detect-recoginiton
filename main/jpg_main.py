# -*- coding: utf-8 -*-import os
import subprocess
import time
import socket
import os
import fcntl
import struct

def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,
        struct.pack('256s', ifname[:15]))[20:24])
while 1:
    try:
        host = get_ip_address('eth0')
    except IOError:
        time.sleep(3)
        continue
    else:
        time.sleep(5)
        break
'''
def isNetOK(testserver):
  s=socket.socket()
  s.settimeout(3)
  try:
    status = s.connect_ex(testserver)
    if status == 0:
      s.close()
      return True
    else:
      return False
  except Exception as e:
    return False
 
def isNetChainOK(testserver=('www.baidu.com',443)):
  isOK = isNetOK(testserver)
  return isOK
while 1: 
    chinanet = isNetChainOK()
    if chinanet:
        break
    else:
        time.sleep(3)
        continue
        '''

#time.sleep(100)
# start capture
os.system("gnome-terminal -e 'bash -c \"cd;cd workspace/live/testProgs/;cktool -s 3 -w 1280 -h 720 -r 1000 -g 12 -o test.264; exec bash\"'")
# strat transfer video
os.system("gnome-terminal -e 'bash -c \"cd;cd workspace/live/testProgs/;./testH264VideoStreamer; exec bash\"'")
# start socket.server
##time.sleep(5)
os.system("gnome-terminal -e 'bash -c \"cd;cd workspace/main/;python s.py; exec bash\"'")
os.system("gnome-terminal -e 'bash -c \"cd;cd workspace/main/;python run_sdk2_cutimage.py; exec bash\"'")
os.system("gnome-terminal -e 'bash -c \"cd;cd workspace/main/;python services.py; exec bash\"'")
# if save_image
