import os
from PIL import Image
import numpy as np
from torchvision import transforms

voc_path='./VOCdevkit/VOC2007'
classes=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable',
         'dog','horse','motorbike','person','pottedplant',
        'sheep','sofa','tvmotor','train','background']

#get color Palette
target=Image.open('./VOCdevkit/VOC2007/SegmentationObject/000042.png')
palette=target.getpalette()
palette=np.reshape(palette,(-1,3)).tolist()
color_map=palette[:21]
num_classes=len(classes)

#read images
def read_images(path=voc_path,train=True):
    #train: boolean value, if true, train file
    if train:
        file=path+'/ImageSets/Segmentation/train.txt'
    else:
        file=path+'/ImageSets/Segmentation/val.txt'
    with open(file) as f:
        imgs=f.read().split()
    img_dir=[path+'/JPEGImages/%s.jpg' %img for img in imgs]
    label_dir=[path+'/SegmentationClass/%s.png' %img for img in imgs]
    return img_dir,label_dir
#crop images to the same size
def crop(data,label,h,w):
    box=[0,0,w,h]
    data=data.crop(box)
    label=label.crop(box)
    return data,label

#get the label of each pixel
cm2lbl=np.zeros(256**3)
for i,cm in enumerate(color_map):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]]=i
def img2label(im): #background:0,aeroplane:1,bicycle:2 ...
    data=np.array(im,dtype='int32')
    idx=(data[:,:,0]*256+data[:,:,1])*256+data[:,:,2]
    return np.array(cm2lbl[idx],dtype='int64')

#image transform
def image_transforms(data,label,height,width):
    data,label=crop(data,label,height,width)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    data=transform(data)
    label=img2label(label)
    return data,label