from torch_geometric.data import DataLoader
from model import *
import random
from tqdm import tqdm
from rotation import augmentation
from torch import optim
import numpy as np
from arg_helper import *

args = parse_arguments()
config = get_config(args.config_file)

seed = config.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# data = torch.load('sulc')
# random.shuffle(data)
#
# train_set = data[0:25]
# valid_set = data[25:29]
# test_set = data[29:]

data = torch.load('mind')
random.shuffle(data)
train_set = data[0:71]
valid_set = data[71:81]
test_set = data[81:]

train_loader = DataLoader(train_set, batch_size = 1, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size = 1)
test_loader = DataLoader(test_set, batch_size = 1)

device = config.device

if config.model == 'edge':
    model = EdgeUnet(config).to(device)
elif config.model == 'transform':
    model = TransUnet(config).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400], gamma=0.8)

train_loss_history=[]
valid_loss_history=[]
best_loss = 10e10
for epoch in tqdm(range(500)):
    model.train()
    train_loss = 0
    valid_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        y = data.y
        weight = torch.bincount(y) / len(y)
        weight = 1 / weight
        weight = weight / weight.sum()
        criterion = torch.nn.CrossEntropyLoss(weight=weight)

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss

    model.eval()
    with torch.no_grad():
        for data in valid_loader:
            data = data.to(device)
            out = model(data)

            y = data.y
            weight = torch.bincount(y) / len(y)
            weight = 1 / weight
            weight = weight / weight.sum()
            criterion = torch.nn.CrossEntropyLoss(weight=weight)

            loss = criterion(out, y)
            valid_loss += loss

    train_loss = train_loss / len(train_set)
    valid_loss = valid_loss / len(valid_set)

    train_loss_history.append(train_loss)
    valid_loss_history.append(valid_loss)

    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), os.path.join(config.save_dir, 'best_model.pt'))

    print(f'Epoch: {epoch:03d} Train Loss: {train_loss:.4f}  Valid Loss: {valid_loss:.4f}')

torch.save(train_loss_history, os.path.join(config.save_dir, 'train_loss'))
torch.save(valid_loss_history, os.path.join(config.save_dir, 'valid_loss'))

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
# print('4')
# print(test_dice)