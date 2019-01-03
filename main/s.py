#!/usr/bin/python27
#coding:utf-8

import SocketServer
import time
import os
import socket
import fcntl
import struct

filepath = '/home/pi/workspace/main/out_allima/'
fn_list = []


def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,
        struct.pack('256s', ifname[:15]))[20:24])


class MyTcpServer(SocketServer.BaseRequestHandler):                           
    def sendfile(self, filename):
        if os.path.isfile(filename):
            print("starting send file <%s> ..." % filename)
            #self.request.send('ready')
            #time.sleep(1)
            f = open(filename, 'rb')
            while True:
                data = f.read(8192)
                if not data:
                    break
                self.request.send(data)
            f.close()
            time.sleep(0.1)
            self.request.send('EOF')
            print("send file success!")
        else:
            print("the file <%s> is not exist" % filename)
            #self.request.send("the file <%s> is not exist" % filename)
                               
    def handle(self):
        print("get connection from :",self.client_address)
        while True:

            try:
                data = self.request.recv(2048)
                print(data)
                if not data:
                    print("break the connection!")
                    break
                elif data == 'Hi, server':   
                    self.request.sendall('hi, client')
                elif data == '1':
                    while True:
                        #self.request.sendall(str(len(fn_list)))
                        fl = self.request.recv(1024)
                        print(fl)
                        if fl == 'Ready':
                            self.request.sendall('ok')
                            while True:
                                for i,j,filename_list in os.walk(filepath):
                                    print(filename_list)
                                    fn_list = filename_list
                                if len(fn_list) > 1:
                                    for f in fn_list:
                                        fn = filepath + f
                                        if os.path.isfile(fn):
                                            self.request.send(fn)
                                            time.sleep(0.6)
                                            self.sendfile(fn)
                                            time.sleep(0.1)
                                            os.remove(fn)      
                                    self.request.sendall(filepath)
                                    print('exit')
                                    break
                                    
                                else:
                                    time.sleep(1)
                                    continue
            except Exception,e:
                print("get error at:",e)
                                            
                                        
if __name__ == "__main__":
    #host = '192.168.1.11'
    host = get_ip_address('eth0')
    print(host)
    port = 6000
    s = SocketServer.ThreadingTCPServer((host,port), MyTcpServer)
    s.serve_forever()
