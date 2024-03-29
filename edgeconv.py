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
from utils import *
from model import Net

data_path = 'sphere5_no_std'
conv = 'edge'
# save_dir = os.path.join('exp',data_path,conv)
#
# os.makedirs(save_dir, exist_ok=True)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

set_seed(1)

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
    mu = torch.mean(data[i].x, 0)
    sigma = torch.sqrt(torch.mean((data[i].x - mu) ** 2, 0))
    data[i].x = (data[i].x - mu) / sigma

    # data[i].x = (data[i].x-data[i].x.min(0).values)/(data[i].x.max(0).values-data[i].x.min(0).values)

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

class Net(torch.nn.Module):
    def __init__(self, in_channels,hidden_channels,out_channels, conv, aggr='max'):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.conv = conv

        if self.conv == 'edge':
            self.conv1 = EdgeConv(nn.Linear(2 * in_channels, self.hidden_channels[0]), aggr)
            self.conv2 = EdgeConv(nn.Linear(2 * (in_channels+self.hidden_channels[0]), self.hidden_channels[1]), aggr)
            self.conv3 = EdgeConv(nn.Linear(2 * (in_channels+self.hidden_channels[0]+self.hidden_channels[1]), self.hidden_channels[2]), aggr)
            self.conv4 = EdgeConv(nn.Linear(2 * (in_channels+self.hidden_channels[0]+self.hidden_channels[1]+self.hidden_channels[2]), self.hidden_channels[3]), aggr)
        else:
            self.conv1 = PointTransformerConv(in_channels, self.hidden_channels[0])
            self.conv2 = PointTransformerConv(in_channels+self.hidden_channels[0], self.hidden_channels[1])
            self.conv3 = PointTransformerConv(in_channels+self.hidden_channels[0]+self.hidden_channels[1], self.hidden_channels[2])
            self.conv4 = PointTransformerConv(in_channels+self.hidden_channels[0]+self.hidden_channels[1]+self.hidden_channels[2], self.hidden_channels[3])


        self.mlp1 = nn.Linear(in_channels+sum(self.hidden_channels), 64)
        self.mlp2 = nn.Linear(64, out_channels)

        self.dropout = nn.Dropout(p = 0.5)

    def forward(self, data):
        x, edge_index, pos, batch = data.x[:,:self.in_channels], data.edge_index, data.x[:,:3], data.batch

        if self.conv == 'edge':
            x1 = self.conv1(x, edge_index)
        else:
            x1 = self.conv1(x, pos, edge_index)
        x1 = F.leaky_relu(x1)
        self.dropout(x1)

        x2 = torch.cat([x, x1], 1)
        if self.conv == 'edge':
            x2 = self.conv2(x2, edge_index)
        else:
            x2 = self.conv2(x2, pos, edge_index)
        x2 = F.leaky_relu(x2)
        self.dropout(x2)

        x3 = torch.cat([x, x1, x2], 1)
        if self.conv == 'edge':
            x3 = self.conv3(x3, edge_index)
        else:
            x3 = self.conv3(x3, pos, edge_index)
        x3 = F.leaky_relu(x3)
        self.dropout(x3)

        x4 = torch.cat([x, x1, x2, x3], 1)
        if self.conv == 'edge':
            x4 = self.conv4(x4, edge_index)
        else:
            x4 = self.conv4(x4, pos, edge_index)
        x4 = F.leaky_relu(x4)

        out = torch.cat([x, x1, x2, x3, x4], 1)
        # m = self.mlp(out)
        # m = m.max(0).values.repeat(len(x), 1)
        #
        # out = torch.cat([out, m], 1)
        out = self.dropout(out)
        out = self.mlp1(out)
        out = self.mlp2(out)

        return out


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
print('drop50')
