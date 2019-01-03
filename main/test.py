# -*- coding: utf-8 -*-
#! /usr/bin/env python3

from mvnc import mvncapi as mvnc
import numpy
import cv2
import sys
import time
import os
import shutil

goal_path = '/home/sdu/main/out_allima/'
jpath = '/home/sdu/main/tmp_jpg/'
tmpth = '/home/sdu/main/tmp_ncs/'

LABELS=('background','cups','prisoner','numbers')

def run_inference(image_to_classify, ssd_mobilenet_graph,fifoIn, fifoOut, fna):

    resized_image = preprocess_image(image_to_classify)
    ssd_mobilenet_graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, resized_image.astype(numpy.float32), None)
    output, userobj = fifoOut.read_elem()


    num_valid_boxes = int(output[0])
    print('total num boxes: ' + str(num_valid_boxes))
    
    image_count = 0
    for box_index in range(num_valid_boxes):
            prison_num=0
            base_index = 7+ box_index * 7
            if (not numpy.isfinite(output[base_index]) or
                    not numpy.isfinite(output[base_index + 1]) or
                    not numpy.isfinite(output[base_index + 2]) or
                    not numpy.isfinite(output[base_index + 3]) or
                    not numpy.isfinite(output[base_index + 4]) or
                    not numpy.isfinite(output[base_index + 5]) or
                    not numpy.isfinite(output[base_index + 6])):
                print('box at index: ' + str(box_index) + ' has nonfinite data, ignoring it')
                continue
            if (output[base_index+1] and output[base_index+2 >40]) :
                prison_num=prison_num+1
            
            x1 = max(0, int(output[base_index + 3] * image_to_classify.shape[0]))
            y1 = max(0, int(output[base_index + 4] * image_to_classify.shape[1]))
            x2 = min(image_to_classify.shape[0], int(output[base_index + 5] * image_to_classify.shape[0]))
            y2 = min(image_to_classify.shape[1], int(output[base_index + 6] * image_to_classify.shape[1]))

            x1_ = str(x1)
            y1_ = str(y1)
            x2_ = str(x2)
            y2_ = str(y2)

            print('box at index: ' + str(box_index) + ' : ClassID: ' + LABELS[int(output[base_index + 1])] + '  '
                  'Confidence: ' + str(output[base_index + 2]*100) + '%  ' +
                  'Top Left: (' + x1_ + ', ' + y1_ + ')  Bottombuf Right: (' + x2_ + ', ' + y2_ + ')')
            imwrite_cut_img(image_to_classify, output[base_index:base_index + 7], image_count, fna)
            image_count += 1
    return prison_num



def preprocess_image(src):

    # scale the image
    NETWORK_WIDTH = 300
    NETWORK_HEIGHT = 300
    img = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

    # adjust values to range between -1.0 and + 1.0
    img = img - 127.5
    img = img * 0.007843
    return img
def main():
    cv_window_name = "SSD MobileNet - hit any key to exit"

    devices = mvnc.enumerate_devices()
    if len(devices) == 0:
        print('No devices found')
        quit()

    device = mvnc.Device(devices[0])
    device.open()
    graph_file_name = 'test.graph'

    with open(graph_file_name, mode='rb') as f:
        graph_in_memory = f.read()


    graph = mvnc.Graph(graph_file_name)
    fifoIn, fifoOut = graph.allocate_with_fifos(device, graph_in_memory)

    # read the image to run an inference on from the disk
    while True:
        f_list = []
        for i,j,filename_list in os.walk(jpath):
            f_list = filename_list
        si1 = len(f_list)
        time.sleep(1.2)
        f9_list = []
        for i,j,filename99_list in os.walk(jpath):
            f9_list = filename99_list
        si2 = len(f9_list)
        if si1 == si2:
            for f in f_list:
                temjpg= jpath + f
                ncsjpg = tmpth + f
                if os.path.isfile(temjpg):
                    mymovefile(temjpg,ncsjpg)
                    infer_image = cv2.imread(ncsjpg)
                    nn_list = f.split('.')
                    nf = nn_list[0]
                    try:
                        run_inference(infer_image, graph,fifoIn, fifoOut, nf)
                    except:
                        os.remove(ncsjpg)
                        continue
                    else:
                        f1_list = []
                        for e,r,filename2_list in os.walk(tmpth):
                            f1_list = filename2_list
                        if len(f1_list) > 1:
                            for ff in f1_list:
                                srcfile = tmpth + ff  
                                dstfile = goal_path + ff
                                mymovefile(srcfile, dstfile)
                        else:
                            os.remove(ncsjpg)
                else:
                    continue
        else:
            continue
        
    # display the results and wait for user to hit a key
    #cv2.imshow(cv_window_name, infer_image)
    #cv2.imwrite("./0833/")
    #cv2.waitKey(0)

    # Clean up the graph and the device
    graph.destroy()
    fifoIn.destroy()
    fifoOut.destroy()
    device.close()

    
