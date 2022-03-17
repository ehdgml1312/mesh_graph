import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from tqdm import tqdm
from torch import optim
import random
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('default')
import torch.nn as nn

np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

# data = torch.load('sulc')
# random.shuffle(data)
# train_set = data[0:25]
# valid_set = data[25:29]
# test_set = data[29:]

data = torch.load('mind')
random.shuffle(data)
train_set = data[0:71]
valid_set = data[71:81]
test_set = data[81:]

train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=1)
test_loader = DataLoader(test_set, batch_size=1)


import torch

from torch_geometric.nn import EdgeConv, DynamicEdgeConv
from point_trans import PointTransformerConv


class Net(torch.nn.Module):
    def __init__(self, out_channels, aggr='max'):
        super().__init__()


        # self.conv1 = EdgeConv(nn.Linear(2 * 6, 64), aggr)
        # self.conv2 = EdgeConv(nn.Linear(2 * 64, 64), aggr)
        # self.conv3 = EdgeConv(nn.Linear(2 * 64, 64), aggr)
        # self.conv4 = EdgeConv(nn.Linear(2 * 64, 64), aggr)
        self.conv1 = PointTransformerConv(6, 64)
        self.conv2 = PointTransformerConv(6 + 64, 64)
        self.conv3 = PointTransformerConv(6 + 64*2, 64)
        self.conv4 = PointTransformerConv(6 + 64*3, 64)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

        self.mlp1 = nn.Linear(6 + 4 * 64 , 64)
        self.mlp2 = nn.Linear(64, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x[:,:], data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)

        x2 = torch.cat([x, x1], 1)
        x2 = self.conv2(x2, edge_index)
        x2 = F.leaky_relu(x2)

        x3 = torch.cat([x, x1, x2], 1)
        x3 = self.conv3(x3, edge_index)
        x3 = F.leaky_relu(x3)

        x4 = torch.cat([x, x1, x2, x3], 1)
        x4 = self.conv4(x4, edge_index)
        x4 = F.leaky_relu(x4)

        out = torch.cat([x, x1, x2, x3, x4], 1)
        # m = self.mlp(out)
        # m = m.max(0).values.repeat(len(x), 1)
        #
        # out = torch.cat([out, m], 1)
        out = self.dropout1(out)
        out = self.mlp1(out)
        out = self.dropout2(out)
        out = self.mlp2(out)

        return out



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = Net(32).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20)

train_loss_history=[]
valid_loss_history=[]
best_loss = 10e10
for epoch in tqdm(range(600)):
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
        torch.save(model.state_dict(), 'exp/mindboggle/trans/best_model')

    print(f'Epoch: {epoch:03d} Train Loss: {train_loss:.4f}  Valid Loss: {valid_loss:.4f}')
torch.save(train_loss_history, 'exp/mindboggle/trans/train_loss.txt')
torch.save(valid_loss_history, 'exp/mindboggle/trans/valid_loss.txt')
#
# def dice(pred, gt):
#     XnY = torch.ones((len(gt))).to(device) * 14
#     for i in range(len(gt)):
#         if pred[i] == gt[i]:
#             XnY[i] = pred[i]
#     D = torch.zeros((14))
#     for j in range(14):
#         if (len(torch.where(pred == j)[0]) + len(torch.where(gt == j)[0])) == 0:
#             D[j] = 0
#         else:
#             D[j] = ((2 * len(torch.where(XnY == j)[0])) / (
#                         len(torch.where(pred == j)[0]) + len(torch.where(gt == j)[0])))
#
#     dice = (torch.sum(D) - D[0]) / 13
#     return dice
#
# model.load_state_dict(best_state)
#
# model.eval()
# with torch.no_grad():
#     test_dice = 0
#
#     for data in test_loader:
#         data = data.to(device)
#         out = model(data)
#         pred = out.argmax(dim=1)
#
#         D = dice(pred, data.y)
#
#         test_dice += D
#     test_dice /= 7
# print(test_dice)
