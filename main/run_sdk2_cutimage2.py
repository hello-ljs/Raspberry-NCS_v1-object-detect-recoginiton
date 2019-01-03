# -*- coding: utf-8 -*-
#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.


from mvnc import mvncapi as mvnc
import numpy
import cv2
import sys
import time
import os
import shutil

dim=(300,300)

goal_path = '/home/pi/workspace/main/out_allima/'
jpath = '/home/pi/workspace/main/tmp_jpg/'
tmpth = '/home/pi/workspace/main/tmp_ncs/'
# ***************************************************************
# Labels for the classifications for the network.
# ***************************************************************
#LABELS = ('background',
#          'face','plate')
LABELS = ('background','people','vehicle','mechanical','human','plate')


def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print "%s not exist!"%(srcfile)
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)           #移动文件
        print("move %s -> %s"%( srcfile,dstfile))
        

# Run an inference on the passed image
# image_to_classify is the image on which an inference will be performed
#    upon successful return this image will be overlayed with boxes
#    and labels identifying the found objects within the image.
# ssd_mobilenet_graph is the Graph object from the NCAPI which will
#    be used to peform the inference.
def run_inference(image_to_classify, ssd_mobilenet_graph,fifoIn, fifoOut, fna):

    # get a resized version of the image that is the dimensions
    # SSD Mobile net expects
    resized_image = preprocess_image(image_to_classify)

    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    ssd_mobilenet_graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, resized_image.astype(numpy.float32), None)

    # ***************************************************************
    # Get the result from the NCS
    # ***************************************************************
    output, userobj = fifoOut.read_elem()

    #   a.	First fp16 value holds the number of valid detections = num_valid.
    #   b.	The next 6 values are unused.
    #   c.	The next (7 * num_valid) values contain the valid detections data
    #       Each group of 7 values will describe an object/box These 7 values in order.
    #       The values are:
    #         0: image_id (always 0)
    #         1: class_id (this is an index into labels)
    #         2: score (this is the probability for the class)
    #         3: box left location within image as number between 0.0 and 1.0
    #         4: box top location within image as number between 0.0 and 1.0
    #         5: box right location within image as number between 0.0 and 1.0
    #         6: box bottom location within image as number between 0.0 and 1.0

    # number of boxes returned
    num_valid_boxes = int(output[0])
    print('total num boxes: ' + str(num_valid_boxes))
    image_count = 0
    for box_index in range(num_valid_boxes):
            base_index = 7+ box_index * 7
            if (not numpy.isfinite(output[base_index]) or
                    not numpy.isfinite(output[base_index + 1]) or
                    not numpy.isfinite(output[base_index + 2]) or
                    not numpy.isfinite(output[base_index + 3]) or
                    not numpy.isfinite(output[base_index + 4]) or
                    not numpy.isfinite(output[base_index + 5]) or
                    not numpy.isfinite(output[base_index + 6])):
                # boxes with non infinite (inf, nan, etc) numbers must be ignored
                print('box at index: ' + str(box_index) + ' has nonfinite data, ignoring it')
                continue

            # clip the boxes to the image size incase network returns boxes outside of the image
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

            # overlay boxes and labels on the original image to classify
            # overlay_on_image(image_to_classify, output[base_index:base_index + 7])

# overlays the boxes and labels onto the display image.
# display_image is the image on which to overlay the boxes/labels
# object_info is a list of 7 values as returned from the network
#     These 7 values describe the object found and they are:
#         0: image_id (always 0 for myriad)
#         1: class_id (this is an index into labels)
#         2: score (this is the probability for the class)
#         3: box left location within image as number between 0.0 and 1.0
#         4: box top location within image as number between 0.0 and 1.0
#         5: box right location within image as number between 0.0 and 1.0
#         6: box bottom location within image as number between 0.0 and 1.0
# returns None
'''
def overlay_on_image(display_image, object_info):

    # the minimal score for a box to be shown
    min_score_percent = 40

    source_image_width = display_image.shape[1]
    source_image_height = display_image.shape[0]

    base_index = 0
    class_id = object_info[base_index + 1]
    percentage = int(object_info[base_index + 2] * 100)
    if (percentage <= min_score_percent):
        # ignore boxes less than the minimum score
        return

    label_text = LABELS[int(class_id)] + " (" + str(percentage) + "%)"
    box_left = int(object_info[base_index + 3] * source_image_width)
    box_top = int(object_info[base_index + 4] * source_image_height)
    box_right = int(object_info[base_index + 5] * source_image_width)
    box_bottom = int(object_info[base_index + 6] * source_image_height)

    box_color = (255, 128, 0)  # box color
    box_thickness = 2
    cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

    # draw the classification label string just above and to the left of the rectangle
    label_background_color = (125, 175, 75)
    label_text_color = (255, 255, 255)  # white text

    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    label_left = box_left
    label_top = box_top - label_size[1]
    if (label_top < 1):
        label_top = 1
    label_right = label_left + label_size[0]
    label_bottom = label_top + label_size[1]
    cv2.rectangle(display_image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                  label_background_color, -1)

    # label text above the box
    cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)
'''
def imwrite_cut_img(display_image, object_info, c, fna):
    # the minimal score for a box to be shown
    min_score_percent = 60 # confidence

    source_image_width = display_image.shape[1]
    source_image_height = display_image.shape[0]

    base_index = 0
    class_id = object_info[base_index + 1]
    percentage = int(object_info[base_index + 2] * 100)
    if (percentage <= min_score_percent):
        # ignore boxes less than the minimum score
        return

    label = LABELS[int(class_id)]
    '''
    box_left = int(0.8*object_info[base_index + 3] * source_image_width)
    box_top = int(0.8*object_info[base_index + 4] * source_image_height)
    box_right = int(1.2*object_info[base_index + 5] * source_image_width)
    box_bottom = int(1.2*object_info[base_index + 6] * source_image_height)'''
    ##############################################################################################################################################################################################################################################
    if (object_info[base_index + 1]!=1):#if the object is not face ,break  face is 1,cups is 2 
       return 
     


    
    box_left = int(object_info[base_index + 3] * source_image_width)
    box_top = int(object_info[base_index + 4] * source_image_height)
    box_right = int(object_info[base_index + 5] * source_image_width)
    box_bottom = int(object_info[base_index + 6] * source_image_height)

    xx1 = max(0, box_left)
    yy1 = max(0, box_top)
    xx2 = min(source_image_width, box_right)
    yy2 = min(source_image_height, box_bottom)

    # roi_img=display_image[xx1:xx2,yy1:yy2]
    roi_img = display_image[yy1:yy2, xx1:xx2]

    #now_time = datetime.now()
    #now_str = datetime.strftime(now_time, '%Y%m%d%H%M%S')
    # global a
    # outimg_path='./out_img_3/'+now_str+'_%s_%s'%(label,str(c))###################################gai
    # outimg_path='/home/pi/workspace/ncsdk-ncsdk2/examples/caffe/out_img_3'+now_str+'_%s_%s'%(label,str(c))
    # cv2.imwrite(outimg_path+'.jpg',roi_img)
    name = tmpth + fna + str(c) + '.jpg'
    cv2.imwrite(name, roi_img)

    #cv2.waitKey(5)
# create a preprocessed image from the source image that complies to the
# network expectations and return it
def preprocess_image(src):

    # scale the image
    NETWORK_WIDTH = 300
    NETWORK_HEIGHT = 300
    img = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

    # adjust values to range between -1.0 and + 1.0
    img = img - 127.5
    img = img * 0.007843
    return img


# This function is called from the entry point to do
# all the work of the program
def main():
    # name of the opencv window
    cv_window_name = "SSD MobileNet - hit any key to exit"

    # Get a list of ALL the sticks that are plugged in
    # we need at least one
    devices = mvnc.enumerate_devices()
    if len(devices) == 0:
        print('No devices found')
        quit()

    # Pick the first stick to run the network
    device = mvnc.Device(devices[0])
    # Open the NCS
    device.open()
    # The graph file that was created with the ncsdk compiler
    graph_file_name = 'test.graph'

    # read in the graph file to memory bufferlobal name 'datetime' is not defined
#pi@raspberrypi:~/workspace/get_face $
    with open(graph_file_name, mode='rb') as f:
        graph_in_memory = f.read()

    # create the NCAPI graph instance from the memory buffer containing the graph file.
    #graph = device.AllocateGraph(graph_in_memory)

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

# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    main()
