#label_pred: pred classification label
#label_true: true label
#n_class:21
'''
confusion matrix:
2d-array, can be written by hist[label_true][label_pred]
the number of row_index predicted to be col_index
e.g.: hist[0][1]=类别为1的像素点被预测成类别为0的数量
diag(hist)= TP
n_class*label_true[mask].astype(int)+label_pred[mask]=二维数组变为一维数组时的地址取值（每个元素大小为1，返回一个numpy 的 list
precision=TP/(TP+FP),Recall=TP/(TP+FN),F1=(2*P+R)/(P+R)
'''
import torch
def _fast_hist(label_true,label_pred,n_class):
    mask=(label_true>=0) & (label_true<n_class)
    confusion_matrix=np.bincount(n_class*label_true[mask].astype(int)+label_pred[mask],minlength=n_class**2).reshape(n_class,n_class)
    return confusion_matrix
def score(label_true,label_pred,n_class):
    hist=np.zeros((n_class,n_class))
    for lt,lp in zip(label_true,label_pred):
        hist+=_fast_hist(lt.flatten(),lp.flatten(),n_class)
    #pixel acc
    acc=np.diag(hist).sum()/hist.sum()
    #MPA: mean pixel acc
    acc_cls=np.diag(hist)/hist.sum(axis=1)
    acc_cls=np.nanmean(acc_cls)
    # MIOU=sum(p_ii/sum(p_ij))/k
    iu=np.diag(hist)/(hist.sum(axis=1)+hist.sum(axis=0)-np.diag(hist)) #TP/(TP+FP+FN)
    mean_iu=np.nanmean(recall)
    #pred frequency
    freq=hist.sum(axis=1)/hist.sum()
    #freq*recall
    fwacc=(freq[freq>0]*recall[freq>0]).sum()
    return acc,acc_cls,mean_iu,fwacc    