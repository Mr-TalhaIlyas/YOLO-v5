import numpy as np
import os, cv2
from tqdm import tqdm, trange
import seaborn as sns

def xyxy_2_xyxyo(img_w, img_h, box):
    '''
    input_box : (xmin, ymin, xmax, ymax) with yolo imge shape 
                default image size for yolo image in 416x320 (w,h) so thats why
    output_box : (xmin, ymin, xmax, ymax) wiht original image shape
    '''
    xmin = (box[0] / 416) 
    ymin = (box[1] / 416)#320 
    xmax = (box[2] / 416)
    ymax = (box[3] / 416) 
    
    box_minmax = np.array([xmin*img_w, ymin*img_h, xmax*img_w, ymax*img_h]).astype(np.int)
    box_minmax[box_minmax<0] = 0 # to make -ve values zero
    return box_minmax
def draw_boxes(image_in, confidences, nms_box, det_classes, classes, order='yx_minmax', analysis=False):
    '''
    Parameters
    ----------
    image : RGB image original shape will be resized
    confidences : confidence scores array, shape (None,)
    nms_box : all the b_box coordinates array after NMS, shape (None, 4) => order [y_min, x_min, y_max, x_max]
    det_classes : shape (None,), names  of classes detected
    classes : all classes names in dataset
    '''
    img_h = 0#416
    img_w = 0#416
    # rescale and resize image
    image = image_in /255
    #image = cv2.resize(image_in, (img_w, img_h))/255
    boxes = np.empty((nms_box.shape))
    if order == 'yx_minmax': # pred
        # form [y_min, x_min, y_max, x_max]  to [x_min, y_min, x_max, y_max]
        # and also making them absolute from relative by mult. wiht img dim.
        boxes[:,1] = nms_box[:,0] * img_h
        boxes[:,0] = nms_box[:,1] * img_w
        boxes[:,3] = nms_box[:,2] * img_h 
        boxes[:,2] = nms_box[:,3] * img_w 
    elif order == 'xy_minmax': # gt
        boxes[:,0] = (nms_box[:,0] )#/ image_in.shape[0] )* img_w
        boxes[:,1] = (nms_box[:,1] )#/ image_in.shape[1] )* img_h
        boxes[:,2] = (nms_box[:,2] )#/ image_in.shape[0] )* img_w 
        boxes[:,3] = (nms_box[:,3] )#/ image_in.shape[1] )* img_h 
    elif order == 'xy_wh': # yolo foramt
        boxes[:,0] = (nms_box[:,0] - (nms_box[:,2] / 2)) * img_w
        boxes[:,1] = (nms_box[:,1] - (nms_box[:,3] / 2)) * img_h
        boxes[:,2] = (nms_box[:,0] + (nms_box[:,2] / 2)) * img_w 
        boxes[:,3] = (nms_box[:,1] + (nms_box[:,3] / 2)) * img_h 
    
    boxes = (boxes).astype(np.uint16)
    i = 1

    colors =  sns.color_palette("Set2") + sns.color_palette("bright")
    [colors.extend(colors) for i in range(3)]
    bb_line_tinkness = 2
    for result in zip(confidences, boxes, det_classes, colors):
        conf = float(result[0])
        facebox = result[1].astype(np.int16)
        #print(facebox)
        name = result[2]
        color = colors[classes.index(name)]#result[3]
        if analysis and order == 'yx_minmax': # pred
            color = (1., 0., 0.) # red  
            bb_line_tinkness = 4
        if analysis and order == 'xy_minmax': # gt
            color = (0., 1., 0.)  # green 
            bb_line_tinkness = 4
        cv2.rectangle(image, (facebox[0], facebox[1]),
                     (facebox[2], facebox[3]), color, bb_line_tinkness)#255, 0, 0
        label = '{0}: {1:0.3f}'.format(name.strip(), conf)
        label_size, base_line = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX   , 0.7, 1)
        
        if not analysis:
            cv2.rectangle(image, (facebox[0], facebox[1] - label_size[1]),    # top left cornor
                         (facebox[0] + label_size[0], facebox[1] + base_line),# bottom right cornor
                         color, cv2.FILLED)
        
            op = cv2.putText(image, label, (facebox[0], facebox[1]),
                       cv2.FONT_HERSHEY_DUPLEX   , 0.7, (0, 0, 0)) 
        i = i+1
    return image#, boxes, det_classes, np.round(confidences, 3)

