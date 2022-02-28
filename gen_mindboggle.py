import torch
from torch_geometric.data import Data, DataLoader
import zipfile
import os

# with zipfile.ZipFile('mindboggle/lh.zip', 'r') as existing_zip:
#     existing_zip.extractall('mindboggle/lh')
# with zipfile.ZipFile('mindboggle/lh_eig.zip', 'r') as existing_zip:
#     existing_zip.extractall('mindboggle/lh_eig')

path1 = 'mindboggle/lh'
path2 = 'mindboggle/lh_eig'

filelist1 = os.listdir(path1)
filelist1.sort()
filelist2 = os.listdir(path2)
filelist2.sort()

data_list=[]

for i in range(101):
    f = open(path1 + '/' + filelist1[2*i],'r')
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

    f1 = open(path1 + '/' + filelist1[2*i+1])
    line1 = f1.read().splitlines()
    x = torch.zeros(num_nodes,6)
    y = torch.zeros(num_nodes,1)
    for j in range(num_nodes):
        feature1 = line1[j].split()
        for k in range(6):
            x[j][k] = float(feature1[k])
        y[j] = float(feature1[6])-1
        y = y.squeeze()
    f1.close()

    mu = torch.mean(x,0)
    sigma = torch.sqrt(torch.mean((x-mu)**2,0))
    x_n = (x-mu)/sigma

    data = Data(x=x_n.float(), edge_index=edge_index,y=y.long())
    data_list.append(data)
torch.save(data_list,'mind')


