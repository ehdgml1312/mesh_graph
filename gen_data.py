import torch
from torch_geometric.data import Data
import os

path = 'sulcus'
filelist = os.listdir(path)
filelist.sort()

data_list=[]

for i in range(36):
    f = open(path + '/' + filelist[2*i],'r')
    line = f.read().splitlines()
    num_nodes = len(line)
    edge_index = torch.ones(1,2,dtype=torch.long)
    for j in range(num_nodes):
        edge = line[j]
        edge = edge.split()
        for k in range(len(edge)):
            a = torch.tensor([[j,int(edge[k])-1]],dtype=torch.long)
            edge_index = torch.cat([edge_index,a],0)
    edge_index = torch.transpose(edge_index[1:],0,1)
    f.close()

    f = open(path + '/' + filelist[2*i+1])
    line = f.read().splitlines()
    x = torch.zeros(num_nodes,4)
    y = torch.zeros(num_nodes,1)
    for j in range(num_nodes):
        feature = line[j].split()
        for k in range(4):
            x[j][k] = float(feature[k])
        if float(feature[-1]) > 10:
            y[j] = float(feature[-1])-2
        else:
            y[j] = float(feature[-1])
        y = y.squeeze()
    f.close()

    mu = torch.mean(x,0)
    sigma = torch.sqrt(torch.mean((x-mu)**2,0))
    x_n = (x-mu)/sigma

    data = Data(x=x_n.float(), edge_index=edge_index,y=y.long())
    data_list.append(data)
torch.save(data_list,'sulc')


