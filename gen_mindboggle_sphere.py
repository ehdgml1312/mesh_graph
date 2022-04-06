import torch
from torch_geometric.data import Data, DataLoader
import zipfile
import os

path1 = 'mindboggle/mesh'

filelist1 = [f for f in os.listdir(path1) if not f.startswith('.')]
filelist1.sort()
filelist1.remove('NKI-TRT-20-6')
filelist1.remove('ico')
filelist1.remove('sphere.sh')
data_list=[]

edge_index = torch.load('mindboggle/mesh/ico/edge_5')

for i in range(100):
    f1 = open(path1 + '/' + filelist1[i] + '/lh.5.x.txt', 'r')
    f2 = open(path1 + '/' + filelist1[i] + '/lh.5.y.txt', 'r')
    f3 = open(path1 + '/' + filelist1[i] + '/lh.5.z.txt', 'r')
    f4 = open(path1 + '/' + filelist1[i] + '/lh.5.curv.txt', 'r')
    f5 = open(path1 + '/' + filelist1[i] + '/lh.5.iH.txt', 'r')
    f6 = open(path1 + '/' + filelist1[i] + '/lh.5.sulc.txt', 'r')
    f7 = open(path1 + '/' + filelist1[i] + '/lh.5.thickness.txt', 'r')
    f8 = open(path1 + '/' + filelist1[i] + '/lh.5.label.txt', 'r')

    line1 = f1.read().splitlines()
    line2 = f2.read().splitlines()
    line3 = f3.read().splitlines()
    line4 = f4.read().splitlines()
    line5 = f5.read().splitlines()
    line6 = f6.read().splitlines()
    line7 = f7.read().splitlines()
    line8 = f8.read().splitlines()

    f1.close(),f2.close(),f3.close(),f4.close(),f5.close(),f6.close(),f7.close(),f8.close()

    line1 = [float(i) for i in line1]
    line2 = [float(i) for i in line2]
    line3 = [float(i) for i in line3]
    line4 = [float(i) for i in line4]
    line5 = [float(i) for i in line5]
    line6 = [float(i) for i in line6]
    line7 = [float(i) for i in line7]
    line8 = [float(i) for i in line8]

    x1,x2,x3,x4,x5,x6,x7,y = torch.tensor(line1),torch.tensor(line2),torch.tensor(line3),torch.tensor(line4),torch.tensor(line5),torch.tensor(line6),torch.tensor(line7),torch.tensor(line8)
    x1,x2,x3,x4,x5,x6,x7 = x1.view(-1,1),x2.view(-1,1),x3.view(-1,1),x4.view(-1,1),x5.view(-1,1),x6.view(-1,1),x7.view(-1,1)

    x = torch.cat((x1,x2,x3,x4,x5,x6,x7),-1)
    y -= 1

    mu = torch.mean(x,0)
    sigma = torch.sqrt(torch.mean((x-mu)**2,0))
    x_n = (x-mu)/sigma

    data = Data(x=x_n.float(), edge_index=edge_index,y=y.long())
    data_list.append(data)
torch.save(data_list,'sphere5')


