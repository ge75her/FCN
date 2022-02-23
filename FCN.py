import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


#init transpose conv kernel
def bilinear_kernel(in_channels,out_channels,kernel_size):
    factor=(kernel_size+1)//2
    if kernel_size%2==1:
        center=factor-1
    else:
        center=factor-0.5
    og=np.ogrid[:kernel_size,:kernel_size]
    filt=(1-abs(og[0]-center)/factor)*(1-abs(og[1]-center)/factor)
    weight=np.zeros((in_channels,out_channels,kernel_size,kernel_size),dtype='float32')
    weight[range(in_channels),range(out_channels),:,:]=filt
    return torch.from_numpy(np.array(weight))


#FCN Model, backbone resnet34
class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        self.backbone = models.resnet34(pretrained=True)
        # [-1]: Linear(512,1000)
        # [-2]:avgpool(1)
        # [-3]:conv(512)*3
        # [-4]:conv(256)*6
        # [:-4]:conv(128) first 2 conv
        self.stage1 = nn.Sequential(*list(self.backbone.children())[:-4])
        self.stage2 = list(self.backbone.children())[-4]
        self.stage3 = list(self.backbone.children())[-3]
        self.conv1 = nn.Conv2d(512, num_classes, 1)
        self.conv2 = nn.Conv2d(256, num_classes, 1)
        self.conv3 = nn.Conv2d(128, num_classes, 1)

        # upsample 8
        self.up8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False)
        self.up8.weight.data = bilinear_kernel(num_classes, num_classes, 16)
        self.up2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.up2.weight.data = bilinear_kernel(num_classes, num_classes, 4)

    def forward(self, x):
        x = self.stage1(x)
        s1 = x  # /8 8 up
        x = self.stage2(x)
        s2 = x  # /16 2up
        x = self.stage3(x)
        s3 = x  # /32 2up FCN32S

        s3 = self.conv1(s3)
        s3 = self.up2(s3)  # /16
        s2 = self.conv2(s2)
        s2 = s2 + s3  # FCN16s
        s1 = self.conv3(s1)
        s2 = self.up2(s2)
        s = s1 + s2  # FCN8S
        s = self.up8(s1)
        return s