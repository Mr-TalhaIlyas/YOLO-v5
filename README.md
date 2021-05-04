# YOLO v5

This repo exaplins how to train [YOLOv5](https://github.com/ultralytics/yolov5) model on your custom dataset. 

## Dependencies

Some of the main requirements are
```
pytorch
```

## Roboflow

I'll be using the [YOLOv5](https://blog.roboflow.com/yolov5-improvements-and-evaluation/) repo in this tutorial, and go through the steps of how 
Original [Colab Notebook](https://colab.research.google.com/drive/1W1-Q37UhxZ99IYKTG1OrsORrKj2R5U17#scrollTo=1NcFxRcFdJ_O)

1. Prepare your data 
2. Convert data format from PASCAl_VOC to YOLO PyTorch format
3. Installing repo

## Dataset Preparation

First to train an object detection model you need a dataset annotated in proper format so download publically available datasets from [here](https://public.roboflow.com/).
I'd recommend starting by downloading already available dataset. There are alot of format options available in Roboflow but for this repo we need `YOLO v5 PyTorch` as this 

![alt text](https://github.com/Mr-TalhaIlyas/YOLO-v5/blob/master/screens/img.png)

or you can also make you own dataset using `labelimg`. A full tutorial for that is [here](https://github.com/tzutalin/labelImg)
The ouput annotation file for label me is `.xml` format but our yolov4 model can't read that so we need to convert the dataset into proper format.
For that follow following steps

## Dataset Format Conversion

After labelling the data put both img files and `.xml` files in the same dir. and run the `voc2yolov5.py` file from scripts

In the first few lines of the `.py` file change following lines accordingly

```python

os.chdir('D:/cw_projects/paprika/paprika_processed/data_final/') # main dir which contains following subdirectories
dirs = ['train', 'test', 'val']

classes = [ "car", "bike","dog","person"] # put your class names
```
After that make your data dir like following

```
📦data
 ┣ 📂test
 ┃ ┣ 📂images
 ┃ ┗ 📂labels
 ┣ 📂train
 ┃ ┣ 📂images
 ┃ ┗ 📂labels
 ┣ 📂valid
 ┃ ┣ 📂images
 ┃ ┗ 📂labels
 ┗ 📜data.yaml
```
the `images` will have all the `.jpg` or `.png` files and `labels` will have `.txt` files for each image and inside `data.yaml` file add following information

```
train: /path2dir/data/train/images
val: /path2dir/data/valid/images

nc: 4
names: [ "car", "bike","dog","person"]
```

## Installation
Getting started wiht the installation make a new conda `env`

```
conda create -n yolo5 python=3.7.6
```
activate `env` and `cd` to a `dir` where you will keep all your `data` and `scripts`.

### YOLO v5
Now form inside the `scirpts` dir copy the `jupyter notebooks` and place in the same `dir` you choose first.

Now run the notebook `setup_YOLOv5.ipynb` sequentelly
Place your data inside the `dir` you choose
Now run the `YOLOv5.ipynd` follow steps inside notebook


