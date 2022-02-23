from data_processing import read_images,crop,img2label,image_transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image

voc_path='./VOCdevkit/VOC2007'
class VOC(Dataset):
    def __init__(self,train,h,w,transform):
        super(VOC,self).__init__()
        self.h=h
        self.w=w
        self.fnum=0 # recorde the filted img
        self.transform=transform
        if train==True:
            data_list,label_list=read_images(path=voc_path,train=True)
        else:
            data_list,label_list=read_images(path=voc_path,train=False)
        self.data_list=self._filter(data_list)
        self.label_list=self._filter(label_list)
    # remove the tiny images
    def _filter(self,images):
        img=[]
        for im in images:
            if (Image.open(im).size[1]>=self.h and Image.open(im).size[0]>=self.w):
                img.append(im)
            else:
                self.fnum=self.fnum+1
        return img
    def __getitem__(self,index):
        img_dir=self.data_list[index]
        label_dir=self.label_list[index]
        image=Image.open(img_dir)
        label=Image.open(label_dir).convert('RGB')
        image,label=self.transform(image,label,self.h,self.w)
        return image,label
    def __len__(self):
        return len(self.data_list)