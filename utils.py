import torch
import random
import numpy as np
import os

def dice(pred, gt):
    XnY = torch.ones((len(gt))).to(device) * 32
    for i in range(len(gt)):
        if pred[i] == gt[i]:
            XnY[i] = pred[i]
    D = torch.zeros((32))
    for j in range(32):
        if (len(torch.where(pred == j)[0]) + len(torch.where(gt == j)[0])) == 0:
            D[j] = 0
        else:
            D[j] = ((2 * len(torch.where(XnY == j)[0])) / (
                        len(torch.where(pred == j)[0]) + len(torch.where(gt == j)[0])))

    dice = (torch.sum(D) - D[0]) /32
    return dice

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)