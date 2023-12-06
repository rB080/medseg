import torch
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve as rc
from sklearn.metrics import auc as auc

def vals(pred, mask):
    pred = torch.round(pred)
    TP = (mask * pred).sum()
    TN = ((1 - mask) * (1 - pred)).sum()
    FP = pred.sum() - TP
    FN = mask.sum() - TP
    return TP, TN, FP, FN


def segmentation_metrics(pred, mask, thresh=0.5):
    TP, TN, FP, FN = vals(pred, mask)
    acc = (TP + TN) / (TP + TN + FP + FN)
    acc = torch.sum(acc).item()
    iou = (TP)/(TP + FN + FP)
    iou = torch.sum(iou).item()
    sen = TP / (TP + FN)
    sen = torch.sum(sen).item()
    prec = (TP) / (TP + FP)
    prec = torch.sum(prec).item()
    recc = TP / (TP + FN)
    recc = torch.sum(recc).item()
    dice = (2*TP)/(2*TP+FP+FN)
    dice = torch.sum(dice).item()
    dict = {"acc": acc, "sen": sen, "pre": prec,
            "rec": recc, "dsc": dice, "iou": iou}
    return dict

def plotROC(GTs, PREDs, save_path, id_text):
    fpr, tpr, _ = rc(GTs, PREDs)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristics, '+id_text)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.5f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(save_path)