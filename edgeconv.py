import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from tqdm import tqdm
from torch import optim
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
from torch_geometric.nn import EdgeConv, DynamicEdgeConv
from point_trans import PointTransformerConv
import copy
from model import Net

data_path = 'mind_no_std'
conv = 'edge'
# save_dir = os.path.join('exp',data_path,conv)
#
# os.makedirs(save_dir, exist_ok=True)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

# data = torch.load('sulc')
# random.shuffle(data)
# train_set = data[0:25]
# valid_set = data[25:29]
# test_set = data[29:]
#
data = torch.load(data_path)
random.shuffle(data)
train_set = data[0:71]
valid_set = data[71:81]
test_set = data[81:]

for i in range(101):
    # mu = torch.mean(data[i].x, 0)
    # sigma = torch.sqrt(torch.mean((data[i].x - mu) ** 2, 0))
    # data[i].x = (data[i].x - mu) / sigma

    data[i].x = (data[i].x-data[i].x.min(0).values)/(data[i].x.max(0).values-data[i].x.min(0).values)

# data = torch.load('sphere6')
# random.shuffle(data)
# train_set = data[0:71]
# valid_set = data[71:81]
# test_set = data[81:]

print(data[0])
print(len(train_set), len(valid_set), len(test_set))
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=1)
test_loader = DataLoader(test_set, batch_size=1)


model = Net(6,[256,128,64,32],32,conv).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20)
print(model)
print(device)
train_loss_history=[]
valid_loss_history=[]
best_loss = 10e10
for epoch in tqdm(range(300)):
    model.train()
    train_loss = 0
    valid_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        weight = torch.bincount(data.y) / len(data.y)
        weight = 1 / weight
        weight = weight / weight.sum()
        criterion = torch.nn.CrossEntropyLoss(weight=weight)

        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        train_loss += loss

    model.eval()
    with torch.no_grad():
        for data in valid_loader:
            data = data.to(device)
            out = model(data)

            weight = torch.bincount(data.y) / len(data.y)
            weight = 1 / weight
            weight = weight / weight.sum()
            criterion = torch.nn.CrossEntropyLoss(weight=weight)

            loss = criterion(out, data.y)
            valid_loss += loss

    train_loss = train_loss / len(train_set)
    valid_loss = valid_loss / len(valid_set)

    train_loss_history.append(train_loss)
    valid_loss_history.append(valid_loss)

    if valid_loss < best_loss:
        best_loss = valid_loss
        best_model = copy.deepcopy(model)
        # torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))

    print(f'Epoch: {epoch:03d} Train Loss: {train_loss:.4f}  Valid Loss: {valid_loss:.4f}')
# torch.save(train_loss_history, os.path.join(save_dir, 'train_loss.txt'))
# torch.save(valid_loss_history, os.path.join(save_dir, 'valid_loss.txt'))
#
def dice(pred, gt):
    XnY = torch.ones((len(gt))).to(device) * 14
    for i in range(len(gt)):
        if pred[i] == gt[i]:
            XnY[i] = pred[i]
    D = torch.zeros((14))
    for j in range(14):
        if (len(torch.where(pred == j)[0]) + len(torch.where(gt == j)[0])) == 0:
            D[j] = 0
        else:
            D[j] = ((2 * len(torch.where(XnY == j)[0])) / (
                        len(torch.where(pred == j)[0]) + len(torch.where(gt == j)[0])))

    dice = (torch.sum(D) - D[0]) / 13
    return dice

#
best_model.eval()
with torch.no_grad():
    test_dice = 0

    for data in test_loader:
        data = data.to(device)
        out = best_model(data)
        pred = out.argmax(dim=1)

        D = dice(pred, data.y)

        test_dice += D
    test_dice /= len(test_set)
print(test_dice)
print('min max')
