from dataLoader import VOC
from FCN import FCN
from confusion_matrix import score
from data_processing import read_images,crop,img2label,image_transforms
from data_processing import classes,num_classes,color_map

from torch.utils.data import Dataset,DataLoader
import torch
from PIL import Image


#train set,val set
h=224
w=224
train_set=VOC(train=True,h=h,w=w,transform=image_transforms)
val_set=VOC(train=False,h=h,w=w,transform=image_transforms)
train_loader=DataLoader(train_set,batch_size=32,shuffle=True)
val_loader=DataLoader(val_set,batch_size=32,shuffle=False)

model = FCN(21)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
cri = torch.nn.NLLLoss()
optim = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-4)
for epoch in range(20):
    train_loss = 0.0
    train_acc=0.0
    train_acc_cls = 0.0
    train_mean_iu = 0.0
    train_fwavacc = 0.0
    model = model.train()
    for i, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        output = torch.nn.functional.log_softmax(output, dim=1)
        loss = cri(output, label)

        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss.item()
        label_pred = output.max(dim=1)[1].data.cpu.numpy()  # index of the biggest output
        label.true = label.data.cpu().numpy()
        for lbt, lbp in zip(label_pred, label_true):
            acc, acc_cls, mean_iu, fwavacc = score(lbt, lbp, num_classes)
            train_acc += acc
            train_acc_cls += acc_cls
            train_mean_iu += mean_iu
            train_fwavacc += fwavacc
    print('epoch', epoch, 'train loss:', train_loss, 'train acc:', train_acc / len(train_set), 'train MIOU:',
          train_mean_iu / len(train_set))

    model = model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_acc_cls=0.0
    val_fwavacc=0.0
    val_mean_iu = 0.0
    for data, label in val_loader:
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        output = torch.nn.functional.log_softmax(output, dim=1)
        loss = cri(output, label)
        val_loss += loss.item()
        labal_pred = output.max(dim=1)[1].data.cpu().numpy()
        label_true = label.data.cpu().numpy()
        for lbt, lbp in zip(label_pred, label_true):
            acc, acc_cls, mean_iu, fwavacc = score(lbt, lbp, num_classes)
            val_acc += acc
            val_acc_cls += acc_cls
            val_mean_iu += mean_iu
            val_fwavacc += fwavacc
    print('val loss:', val_loss, 'val_acc:', val_acc / len(val_set), 'val MIOU:', val_mean_iu / len(val_set))