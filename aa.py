import torch
from torch_sparse import spspmm
import torch.nn.functional as F
import torch.nn as nn
import random
from torch import optim
from tqdm import tqdm
from torch_geometric.data import DataLoader
import copy
import numpy as np
import matplotlib.pyplot as plt
from model import *
from torch_geometric.utils import get_laplacian, to_dense_adj, degree
from pygsp import graphs

plt.style.use('default')
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

    dice = torch.sum(D)/32
    return dice
data = torch.load('sphere5')
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
random.shuffle(data)

train_set = data[0:71]
valid_set = data[71:81]
test_set = data[81:]
print(len(train_set), len(valid_set), len(test_set))

train_loader = DataLoader(train_set, batch_size = 1, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size = 1)
test_loader = DataLoader(test_set, batch_size = 1)
for i in range(101):
    x, edge_index = data[i].x, data[i].edge_index
    A = to_dense_adj(edge_index)[i]
    row, col = data[i].edge_index
    deg = degree(col, data[i].x.size(i), dtype=data[i].x.dtype)
    L = -A
    for j in range(10242):
        L[j][j] = deg[j] + L[j][j]
    G = graphs.Graph(A.numpy())
    U = G.U
    e = G.e
    G.compute_laplacian('combinatorial')
    L = G.L
    break

class spec(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.conv1 = EdgeConv(nn.Linear(2 * in_channels, self.hidden_channels[0])).to('cuda:2')
        self.conv2 = EdgeConv(nn.Linear(2 * self.hidden_channels[0], self.hidden_channels[1])).to('cuda:2')
        self.conv3 = EdgeConv(nn.Linear(2 * self.hidden_channels[1], self.hidden_channels[2])).to('cuda:2')
        self.conv4 = EdgeConv(nn.Linear(2 * self.hidden_channels[2], self.hidden_channels[3])).to('cuda:2')

        self.encode = nn.Linear(1, 7).to('cuda:3')
        self.decode = nn.Linear(128, 32).to('cuda:2')

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, data, u, lamda):
        x0, edge_index = data.x[:, :self.in_channels], data.edge_index

        s0 = self.encode(lamda.to('cuda:3'))

        s1 = torch.matmul(u.transpose(0, 1).to('cuda:3'), s0)
        x1 = self.conv1(x0, edge_index)

        s1 = torch.cat([s1, torch.matmul(u.transpose(0, 1).to('cuda:3'), x0.to('cuda:3'))], 1)
        x1 = torch.cat([x1, torch.matmul(u, s0.to('cuda:2'))], 1)

        s2 = torch.matmul(u.transpose(0, 1).to('cuda:3'), s1)
        x2 = self.conv2(x1, edge_index)

        s2 = torch.cat([s2, torch.matmul(u.transpose(0, 1).to('cuda:3'), x1.to('cuda:3'))], 1)
        x2 = torch.cat([x2, torch.matmul(u, s1.to('cuda:2'))], 1)

        s3 = torch.matmul(u.transpose(0, 1).to('cuda:3'), s2)
        x3 = self.conv3(x2, edge_index)

        s3 = torch.cat([s3, torch.matmul(u.transpose(0, 1).to('cuda:3'), x2.to('cuda:3'))], 1)
        x3 = torch.cat([x3, torch.matmul(u, s2.to('cuda:2'))], 1)

        s4 = torch.matmul(u.transpose(0, 1), s3)
        x4 = self.conv4(x3, edge_index)

        s4 = torch.cat([s4, torch.matmul(u.transpose(0, 1).to('cuda:3'), x3.to('cuda:3'))], 1)
        x4 = torch.cat([x4, torch.matmul(u, s3.to('cuda:2'))], 1)

        # m = self.mlp(out)
        # m = m.max(0).values.repeat(len(x), 1)
        #
        # out = torch.cat([out, m], 1)

        out = self.dropout(x4)
        out = self.decode(out)

        return out

device = 'cuda:2'
model = spec(7,[64,64,64,64],32)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20)

u = torch.tensor(U).to(device)
e = torch.tensor(e).view(-1,1).to(device)

train_loss_history = []
valid_loss_history = []
best_loss = 10e10
for epoch in tqdm(range(300)):
    model.train()
    train_loss = 0
    valid_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data, u, e)

        weight = torch.bincount(data.y) / len(data.y)
        weight = 1 / weight
        weight = weight / weight.sum()

        criterion = torch.nn.CrossEntropyLoss(weight=weight)

        loss = criterion(out, data.y)
        loss.backward

        optimizer.step()

        train_loss += loss

    model.eval()
    with torch.no_grad():
        for data in valid_loader:
            data = data.to(device)

            out = model(data, u, e)

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

    print(f'Epoch: {epoch:03d} Train Loss: {train_loss:.4f}  Valid Loss: {valid_loss:.4f}')

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

