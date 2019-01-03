from mvnc import mvncapi as mvnc
import numpy
import cv2
import sys


IMAGE_FULL_PATH = '/home/pi/workspace/main/out_allima/yuv_tojpg.jpg'
goal_path = '/home/pi/workspace/main/out_allima/'
LABELS = ('background','people','vehicle','mechanical','human','plate')

def run_inference(image_to_classify, ssd_mobilenet_graph,fifoIn, fifoOut):
    resized_image = preprocess_image(image_to_classify)
    ssd_mobilenet_graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, resized_image.astype(numpy.float32), None)
    output, userobj = fifoOut.read_elem()
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
            imwrite_cut_img(image_to_classify, output[base_index:base_index + 7], image_count)
            image_count += 1

            # overlay boxes and labels on the original image to classify
            # overlay_on_image(image_to_classify, output[base_index:base_index + 7])

def imwrite_cut_img(display_image, object_info, c):
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

    box_left = int(object_info[base_index + 3] * source_image_width)
    box_top = int(object_info[base_index + 4] * source_image_height)
    box_right = int(object_info[base_index + 5] * source_image_width)
    box_bottom = int(object_info[base_index + 6] * source_image_height)

    xx1 = max(0, box_left)
    yy1 = max(0, box_top)
    xx2 = min(source_image_width, box_right)
    yy2 = min(source_image_height, box_bottom)

    roi_img = display_image[yy1:yy2, xx1:xx2]


    name = goal_path + str(c) + '.jpg'
    cv2.imwrite(name, roi_img)

    cv2.waitKey(5)
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
    infer_image = cv2.imread(IMAGE_FULL_PATH)
    run_inference(infer_image, graph,fifoIn, fifoOut)
    graph.destroy()
    fifoIn.destroy()
    fifoOut.destroy()
    device.close()

if __name__ == "__main__":
    main()
