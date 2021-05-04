import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
torch.rand(10)
import torch.nn as nn
import torch.nn.functional as F
import glob
from tqdm import tqdm, trange
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
print(torch.cuda.current_device())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
import torch.backends.cudnn as cudnn
import numpy as np
import os, cv2
from tqdm import tqdm, trange
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from my_utils import xyxy_2_xyxyo, draw_boxes
# Initialize
set_logging()
device = select_device('')
half = device.type != 'cpu'  # half precision only supported on CUDA

def prepare_input(img1, img_size=416):
    
    img2 = cv2.resize(img1, (img_size, img_size)) # W x H
    img2 = img2.transpose(2,0,1)
    img2 = img2[np.newaxis, ...]
    img2 = torch.from_numpy(img2).to(device) # torch image is ch x H x W
    img2 = img2.half() if half else img2.float()
    img2 /= 255.0
    
    return img2
#%%
# Directories
save_dir = '/home/user01/data_ssd/Talha/yolo/op/'
weights = '/home/user01/data_ssd/Talha/yolo/yolov5/runs/train/yolov5s_results/weights/best.pt'
source = '/home/user01/data_ssd/Talha/yolo/paprika_y5/test/images/'
imgsz = 416
conf_thres = 0.4
iou_thres = 0.5
classes =  [0,1,2,3,4,5] 
class_names = ["blossom_end_rot", "graymold","powdery_mildew","spider_mite",
               "spotting_disease", "snails_and_slugs"] 

# deleting files in op_dir
filelist = [ f for f in os.listdir(save_dir)]# if f.endswith(".png") ]
for f in tqdm(filelist, desc = 'Deleting old files fro directory'):
    os.remove(os.path.join(save_dir, f))
    

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size
names = model.module.names if hasattr(model, 'module') else model.names  # get class names
if half:
    model.half()  # to FP16


img_paths = glob.glob('/home/user01/data_ssd/Talha/yolo/paprika_y5/test/images/*.png') + \
            glob.glob('/home/user01/data_ssd/Talha/yolo/paprika_y5/test/images/*.jpg')

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
#%%
for i in trange(len(img_paths)):
    
    
    path = img_paths[i]
    img1  = cv2.imread(path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = img1.shape
    
    img2 = prepare_input(img1, img_size=416)
    # get file name
    name = os.path.basename(path)[:-4]
    # Inference
    t1 = time_synchronized()
    pred = model(img2, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=True)
    if pred[0] is not None:
        boxes = pred[0].cpu().numpy() # <xmin><ymin><xmax><ymax><confd><class_id>
    else:
        boxes = np.array([10.0, 20.0, 30.0, 50.0, 0.75, 0]).reshape(1,6) # dummy values
    coords_minmax = np.zeros((boxes.shape[0], 4)) # droping 5th value
    confd = np.zeros((boxes.shape[0], 1))
    class_ids = np.zeros((boxes.shape[0], 1))
    # assign
    coords_minmax = boxes[:,0:4] # coords
    confd = boxes[:,4] # confidence
    class_ids = boxes[:,5] # class id
    
    coords_xyminmax = []
    det_classes = []
    for i in range(boxes.shape[0]):
        coords_xyminmax.append(xyxy_2_xyxyo(img_w, img_h, coords_minmax[i]))
        det_classes.append(class_names[int(class_ids[i])])
    all_bounding_boxnind = []
    for i in range(boxes.shape[0]):
        
        bounding_box = [0.0] * 6
        
        bounding_box[0] = det_classes[i]
        bounding_box[1] = confd[i]
        bounding_box[2] = coords_xyminmax[i][0]
        bounding_box[3] = coords_xyminmax[i][1]
        bounding_box[4] = coords_xyminmax[i][2]
        bounding_box[5] = coords_xyminmax[i][3]
        
        bounding_box = str(bounding_box)[1:-1]# remove square brackets
        bounding_box = bounding_box.replace("'",'')# removing inverted commas around class name
        bounding_box = "".join(bounding_box.split())# remove spaces in between **here dont give space inbetween the inverted commas "".
        all_bounding_boxnind.append(bounding_box)
    
    all_bounding_boxnind = ' '.join(map(str, all_bounding_boxnind))# convert list to string
    all_bounding_boxnind=list(all_bounding_boxnind.split(' ')) # convert strin to list
    # replacing commas with spaces
    for i in range(len(all_bounding_boxnind)):
        all_bounding_boxnind[i] = all_bounding_boxnind[i].replace(',',' ')
    for i in range(len(all_bounding_boxnind)):
    # check if file exiscts else make new
        with open(save_dir +'{}.txt'.format(name), "a+") as file_object:
            # Move read cursor to the start of file.
            file_object.seek(0)
            # If file is not empty then append '\n'
            data = file_object.read(100)
            if len(data) > 0 :
                file_object.write("\n")
            # Append text at the end of file
            file_object.write(all_bounding_boxnind[i])
#%%
import cv2
import glob, random
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

img_paths = glob.glob('/home/user01/data_ssd/Talha/yolo/paprika_y5/valid/images/*.png') + \
            glob.glob('/home/user01/data_ssd/Talha/yolo/paprika_y5/valid/images/*.jpg')
img_path = random.choice(img_paths)

img1  = cv2.imread(img_path)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img_h, img_w, _ = img1.shape

img2 = prepare_input(img1, img_size=416)

pred = model(img2, augment=False)[0]

# Apply NMS
pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=True)
boxes = pred[0].cpu().numpy() # <xmin><ymin><xmax><ymax><confd><class_id>

coords_minmax = np.zeros((boxes.shape[0], 4)) # droping 5th value
confd = np.zeros((boxes.shape[0], 1))
class_ids = np.zeros((boxes.shape[0], 1))
# assign
coords_minmax = boxes[:,0:4] # coords
confd = boxes[:,4] # confidence
class_ids = boxes[:,5] # class id
    
coords_xyminmax = []
det_classes = []
for i in range(boxes.shape[0]):
    coords_xyminmax.append(xyxy_2_xyxyo(img_w, img_h, coords_minmax[i]))
    det_classes.append(class_names[int(class_ids[i])])

t = np.asarray(coords_xyminmax)
op = draw_boxes(img1, confd, t, det_classes, class_names, order='xy_minmax', analysis=False)
plt.imshow(op)
print('='*50)
print('Image Name: ', os.path.basename(img_path),img1.shape)
print('\nClass_name      ', '| B_box Coords ', '| Confidence')
print('_'*50)
for k in range(len(det_classes)):
    print(det_classes[k], t[k], confd[k])
print('='*50)